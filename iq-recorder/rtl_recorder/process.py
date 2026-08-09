"""Cross-platform wrapper around a running ``rtl_sdr`` subprocess.

Stopping a subprocess "cleanly" is more subtle than it looks:

- On POSIX, ``rtl_sdr`` installs a SIGINT/SIGTERM handler that closes the
  output file and releases the USB device before exiting - so we prefer
  SIGINT and only escalate to terminate()/kill() if it doesn't respond.
- On Windows, ``Popen.terminate()`` maps to ``TerminateProcess``, which
  gives the process no chance to run any cleanup at all. That risks a
  truncated IQ file and, more importantly, leaves the USB device claimed
  so the *next* recording fails with a "device busy" error. Instead we
  launch the process in its own console process group and deliver
  CTRL_BREAK_EVENT, which ``rtl_sdr``'s Windows console-control handler
  responds to the same way it responds to Ctrl+C on POSIX.

Either way, a hard kill() is always available as a last resort so we never
hang indefinitely waiting for a misbehaving process.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
from typing import List, Optional

from .utils import is_windows

logger = logging.getLogger(__name__)

#: Cap on how many stderr lines we retain in memory for diagnostics.
_MAX_STDERR_LINES = 500


class ProcessHandle:
    """Launches and controls a single ``rtl_sdr`` child process."""

    def __init__(self, cmd: List[str], cwd: Optional[str] = None):
        self.cmd = cmd
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None
        self._stderr_lines: List[str] = []
        self._stderr_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> int:
        """Launch the process and start pumping its stderr in the background."""
        kwargs = {}
        if is_windows():
            # Required so CTRL_BREAK_EVENT can later be targeted at this
            # process (and not this Python process itself).
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **kwargs,
        )
        self._reader_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._reader_thread.start()
        return self.proc.pid

    def _pump_stderr(self) -> None:
        """Continuously drain stderr so the child never blocks on a full pipe."""
        assert self.proc is not None and self.proc.stderr is not None
        try:
            for line in self.proc.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > _MAX_STDERR_LINES:
                        self._stderr_lines.pop(0)
                logger.debug("rtl_sdr: %s", line)
        except (ValueError, OSError):
            # Pipe closed underneath us during shutdown - not an error.
            pass

    @property
    def stderr_text(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def poll(self) -> Optional[int]:
        """Return the exit code if the process has exited, else None."""
        return self.proc.poll() if self.proc else None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def stop(self, grace_seconds: float = 5.0) -> None:
        """Stop the process, escalating from graceful to forceful.

        Order: platform-appropriate "graceful" signal -> wait -> terminate()
        -> wait -> kill(). Never raises; logs and returns once the process
        is confirmed gone (or we've exhausted our escalation options).
        """
        if self.proc is None or self.proc.poll() is not None:
            return

        try:
            if is_windows():
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                self.proc.send_signal(signal.SIGINT)
        except Exception:
            logger.debug("Graceful stop signal failed for PID %s", self.pid, exc_info=True)

        if self._wait(grace_seconds):
            logger.info("Process %s stopped gracefully", self.pid)
            return

        logger.warning("Process %s did not exit within %.1fs, sending terminate()", self.pid, grace_seconds)
        try:
            self.proc.terminate()
        except Exception:
            logger.debug("terminate() failed for PID %s", self.pid, exc_info=True)
        if self._wait(grace_seconds):
            return

        logger.warning("Process %s still alive, sending kill()", self.pid)
        try:
            self.proc.kill()
        except Exception:
            logger.debug("kill() failed for PID %s", self.pid, exc_info=True)
        if not self._wait(grace_seconds):
            logger.error("Process %s could not be confirmed dead", self.pid)

    def _wait(self, timeout: float) -> bool:
        try:
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
