"""The RTL-SDR Recording Manager.

``RTLSDRRecorder`` is the single public entry point for this subsystem. It
owns the recorder's state machine, launches/monitors/stops the ``rtl_sdr``
child process (or simulates one), verifies the resulting IQ file, and
writes the JSON metadata sidecar. It knows nothing about SatNOGS, pass
scheduling, or signal processing - a future scheduler is expected to call
it through ``record()`` / ``record_pass()`` (Phase 2) or the lower-level
``start_recording()`` / ``stop_recording()`` / ``cancel_recording()`` /
``check_recording_status()`` primitives.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import RecorderConfig, validate_duration, validate_frequency, validate_gain, validate_sample_rate
from .device import DeviceCheckResult
from .device import check_device as _check_device
from .exceptions import (
    DeviceBusyError,
    DeviceNotFoundError,
    OutputDirectoryError,
    RecordingProcessError,
    RTLSDRError,
)
from .filenames import build_base_filename, unique_output_paths
from .metadata import RecordingMetadata, RecordingResult, to_iso
from .process import ProcessHandle
from .states import RecorderState, RecordingStatus
from .utils import check_disk_space, expected_iq_file_size, find_executable

logger = logging.getLogger(__name__)

#: States in which a *different* record() call must be rejected as busy.
_ACTIVE_STATES = (RecorderState.WAITING, RecorderState.STARTING, RecorderState.RECORDING, RecorderState.STOPPING)
_TERMINAL_STATES = (RecorderState.COMPLETED, RecorderState.FAILED, RecorderState.CANCELLED)

_BUSY_MARKERS = ("usb_claim_interface error", "usb_open error", "error claiming", "resource busy")
_NOT_FOUND_MARKERS = ("no supported devices found",)


@dataclass
class _Session:
    """Bookkeeping for a single in-progress (or just-finished) recording attempt."""

    satellite_name: str
    norad_id: Optional[int]
    frequency_hz: int
    sample_rate: int
    gain: Optional[float]
    duration: Optional[float]
    scheduled_aos: Optional[datetime]
    scheduled_los: Optional[datetime]
    pre_buffer_seconds: float
    post_buffer_seconds: float
    output_path: Path
    metadata_path: Path
    command: Optional[List[str]] = None
    process: Optional[ProcessHandle] = None
    actual_start: Optional[datetime] = None
    actual_stop: Optional[datetime] = None
    result: Optional[RecordingResult] = None
    #: Set once finalisation (stop signal + verification + metadata write) is
    #: complete. Lets a second concurrent caller (e.g. stop_recording() from
    #: another thread while record()'s own wait loop is also finalising)
    #: wait for the in-progress finalize instead of racing to do it twice.
    finalized_event: threading.Event = field(default_factory=threading.Event)


class RTLSDRRecorder:
    """Controls one RTL-SDR device and records IQ samples to disk on request."""

    def __init__(self, config: Optional[RecorderConfig] = None):
        self.config = config or RecorderConfig()
        self._lock = threading.RLock()
        self._state = RecorderState.IDLE
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._session: Optional[_Session] = None

    # ------------------------------------------------------------------ #
    # State / device inspection
    # ------------------------------------------------------------------ #

    def _set_state(self, new_state: RecorderState, reason: str = "") -> None:
        old = self._state
        self._state = new_state
        suffix = f" ({reason})" if reason else ""
        logger.info("State transition: %s -> %s%s", old.value, new_state.value, suffix)

    def check_recording_status(self) -> RecorderState:
        """Return the recorder's current lifecycle state."""
        with self._lock:
            return self._state

    def check_device(self) -> DeviceCheckResult:
        """Probe for a connected, available RTL-SDR device (via rtl_test)."""
        if self.config.simulate:
            logger.info("Simulation mode enabled - skipping hardware device check")
            return DeviceCheckResult(True, "Simulation mode - device check skipped")
        result = _check_device(self.config.rtl_test_path)
        if result.available:
            logger.info("RTL-SDR detected: %s", result.message)
        else:
            logger.warning("RTL-SDR device check failed: %s", result.message)
        return result

    # ------------------------------------------------------------------ #
    # Low-level primitives
    # ------------------------------------------------------------------ #

    def start_recording(self, *, satellite_name: str, frequency_hz, sample_rate, gain,
                         duration=None, norad_id: Optional[int] = None,
                         scheduled_aos: Optional[datetime] = None, scheduled_los: Optional[datetime] = None,
                         pre_buffer_seconds: float = 0.0, post_buffer_seconds: float = 0.0,
                         output_dir: Optional[str] = None) -> None:
        """Validate parameters, then launch (or simulate) the recording.

        Returns once the process is confirmed alive (real mode) or the
        placeholder file is written (simulation mode) and the recorder is
        in the RECORDING state. Raises an ``RTLSDRError`` subclass on any
        failure; ``record()`` wraps this and translates failures into a
        ``RecordingResult`` instead of propagating.
        """
        with self._lock:
            if self._state in _ACTIVE_STATES:
                raise DeviceBusyError(
                    f"DEVICE_BUSY: recorder already has an active recording (state={self._state.value})"
                )
            self._cancel_event.clear()
            self._stop_event.clear()
            self._set_state(RecorderState.STARTING, f"record requested for {satellite_name}")

        session: Optional[_Session] = None
        try:
            freq = validate_frequency(frequency_hz)
            rate = validate_sample_rate(sample_rate)
            gain_db = validate_gain(gain)
            dur = validate_duration(duration) if duration is not None else None

            out_dir = Path(output_dir) if output_dir else self.config.resolve_output_dir()
            if output_dir:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise OutputDirectoryError(f"Cannot create/access output directory '{out_dir}': {exc}") from exc
                if not os.access(out_dir, os.W_OK):
                    raise OutputDirectoryError(f"Output directory '{out_dir}' is not writable")

            base_name = build_base_filename(satellite_name, datetime.now(timezone.utc), freq)
            output_path, metadata_path = unique_output_paths(out_dir, base_name)

            if dur is not None:
                expected_size = expected_iq_file_size(rate, dur)
                if not check_disk_space(out_dir, expected_size, self.config.disk_space_margin):
                    raise OutputDirectoryError(
                        f"Not enough free disk space in '{out_dir}' for the expected "
                        f"recording size (~{expected_size} bytes)"
                    )

            session = _Session(
                satellite_name=satellite_name,
                norad_id=norad_id,
                frequency_hz=freq,
                sample_rate=rate,
                gain=gain_db,
                duration=dur,
                scheduled_aos=scheduled_aos,
                scheduled_los=scheduled_los,
                pre_buffer_seconds=pre_buffer_seconds,
                post_buffer_seconds=post_buffer_seconds,
                output_path=output_path,
                metadata_path=metadata_path,
            )

            if self.config.simulate:
                self._start_simulated(session)
            else:
                self._start_real(session)

            with self._lock:
                self._session = session
                self._set_state(RecorderState.RECORDING, f"{satellite_name} -> {output_path.name}")
            logger.info(
                "Recording started: satellite=%s freq=%d sample_rate=%d gain=%s output=%s",
                satellite_name, freq, rate, gain_db, output_path.name,
            )

        except RTLSDRError as exc:
            self._fail_start(session, exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            wrapped = RecordingProcessError(f"Unexpected error starting recording: {exc}")
            self._fail_start(session, wrapped)
            raise wrapped from exc

    def _fail_start(self, session: Optional[_Session], exc: Exception) -> None:
        with self._lock:
            self._session = session
            self._set_state(RecorderState.FAILED, str(exc))
        if session is not None:
            status = RecordingStatus.DEVICE_BUSY if isinstance(exc, DeviceBusyError) else RecordingStatus.FAILED
            self._write_result(session, status, str(exc))
        logger.error("Failed to start recording: %s", exc)

    def _start_simulated(self, session: _Session) -> None:
        logger.info("SIMULATE: pretending to start rtl_sdr for %s", session.satellite_name)
        session.actual_start = datetime.now(timezone.utc)
        if session.duration:
            size = min(expected_iq_file_size(session.sample_rate, session.duration), self.config.simulate_max_bytes)
        else:
            size = self.config.simulate_max_bytes
        session.output_path.write_bytes(os.urandom(max(size, 1)))

    def _start_real(self, session: _Session) -> None:
        exe = find_executable("rtl_sdr", self.config.rtl_sdr_path)
        cmd = [exe, "-d", str(self.config.device_index), "-f", str(session.frequency_hz), "-s", str(session.sample_rate)]
        if session.gain is not None:
            cmd += ["-g", str(session.gain)]
        cmd.append(str(session.output_path))
        session.command = cmd

        process = ProcessHandle(cmd)
        pid = process.start()
        session.process = process
        logger.info("Launched rtl_sdr PID=%s: %s", pid, " ".join(cmd))

        time.sleep(self.config.startup_grace_seconds)
        exit_code = process.poll()
        if exit_code is not None:
            stderr = process.stderr_text
            lowered = stderr.lower()
            if any(marker in lowered for marker in _BUSY_MARKERS):
                raise DeviceBusyError(f"RTL-SDR device busy (rtl_sdr exited {exit_code}): {stderr or '(no stderr)'}")
            if any(marker in lowered for marker in _NOT_FOUND_MARKERS):
                raise DeviceNotFoundError(f"RTL-SDR device not found (rtl_sdr exited {exit_code}): {stderr or '(no stderr)'}")
            raise RecordingProcessError(
                f"rtl_sdr exited immediately (code {exit_code}): {stderr or '(no stderr - check permissions/driver)'}"
            )
        session.actual_start = datetime.now(timezone.utc)

    def stop_recording(self) -> RecordingResult:
        """Gracefully stop an active recording and finalise its result."""
        with self._lock:
            state = self._state
            if state != RecorderState.RECORDING:
                if self._session is not None and self._session.result is not None:
                    return self._session.result
                return RecordingResult(
                    status=RecordingStatus.FAILED,
                    error_message=f"Cannot stop: recorder is in state {state.value}, not RECORDING",
                )
            self._stop_event.set()
        logger.info("Stop requested")
        return self._finalize("stopped")

    def cancel_recording(self) -> RecordingResult:
        """Cancel a waiting/starting/active recording and finalise its result."""
        with self._lock:
            state = self._state
            if state not in (RecorderState.WAITING, RecorderState.STARTING, RecorderState.RECORDING):
                if self._session is not None and self._session.result is not None:
                    return self._session.result
                return RecordingResult(
                    status=RecordingStatus.FAILED, error_message=f"Cannot cancel: recorder is in state {state.value}"
                )
            self._cancel_event.set()
        logger.warning("Cancel requested")
        return self._finalize("cancelled")

    # ------------------------------------------------------------------ #
    # High-level, blocking convenience API
    # ------------------------------------------------------------------ #

    def record(self, *, satellite_name: str, frequency_hz, sample_rate, gain, duration,
               norad_id: Optional[int] = None, output_dir: Optional[str] = None) -> RecordingResult:
        """Record for a fixed duration starting immediately. Blocks until done.

        This is the manual/testing entry point (§5/§15 of the spec). It
        never raises for expected failures - check ``result.status``.
        """
        try:
            self.start_recording(
                satellite_name=satellite_name,
                frequency_hz=frequency_hz,
                sample_rate=sample_rate,
                gain=gain,
                duration=duration,
                norad_id=norad_id,
                output_dir=output_dir,
            )
        except RTLSDRError as exc:
            status = RecordingStatus.DEVICE_BUSY if isinstance(exc, DeviceBusyError) else RecordingStatus.FAILED
            return RecordingResult(status=status, error_message=str(exc))

        reason = self._wait_for_stop_condition(duration)
        return self._finalize(reason)

    def record_pass(self, **kwargs) -> RecordingResult:
        """Scheduled AOS/LOS recording - implemented in Phase 2.

        The intended signature (accepting UTC ``aos``/``los`` datetimes and
        ``pre_buffer``/``post_buffer`` seconds) is documented in the
        project README; it is intentionally not implemented yet so that
        Phase 1 (manual recording + simulation) can be validated first.
        """
        raise NotImplementedError(
            "record_pass() (scheduled AOS/LOS recording) is implemented in Phase 2. "
            "Use record() for manual/immediate recordings in Phase 1."
        )

    # ------------------------------------------------------------------ #
    # Internal: wait loop + finalisation
    # ------------------------------------------------------------------ #

    def _wait_for_stop_condition(self, duration: Optional[float]) -> str:
        """Block until cancel/stop/crash/duration-elapsed, whichever comes first."""
        start = time.monotonic()
        session = self._session
        try:
            while True:
                if self._cancel_event.is_set():
                    return "cancelled"
                if self._stop_event.is_set():
                    return "stopped"
                if not self.config.simulate and session is not None and session.process is not None:
                    if session.process.poll() is not None:
                        return "crashed"
                elapsed = time.monotonic() - start
                if duration is not None and elapsed >= duration:
                    return "duration_elapsed"
                sleep_for = self.config.monitor_interval_seconds
                if duration is not None:
                    sleep_for = max(0.01, min(sleep_for, duration - elapsed))
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received - cancelling recording")
            self._cancel_event.set()
            return "cancelled"

    def _finalize(self, reason: str) -> RecordingResult:
        with self._lock:
            session = self._session
            if session is None:
                return RecordingResult(status=RecordingStatus.FAILED, error_message="No recording session to finalize")
            if self._state in _TERMINAL_STATES:
                return session.result  # idempotent - already finalised
            if self._state == RecorderState.STOPPING:
                # Another thread (e.g. record()'s own wait loop, racing
                # against an external stop_recording()/cancel_recording()
                # call) is already finalising this session - wait for it
                # rather than doing the stop/verify/write work twice.
                already_finalizing = True
            else:
                already_finalizing = False
                self._set_state(RecorderState.STOPPING, reason)

        if already_finalizing:
            session.finalized_event.wait(timeout=self.config.stop_grace_seconds + 10)
            return session.result

        if not self.config.simulate and session.process is not None:
            session.process.stop(self.config.stop_grace_seconds)

        session.actual_stop = datetime.now(timezone.utc)

        if reason == "cancelled":
            status, error_message = RecordingStatus.CANCELLED, "Recording cancelled"
        elif reason == "crashed":
            stderr = session.process.stderr_text if session.process else ""
            status = RecordingStatus.FAILED
            error_message = f"rtl_sdr process terminated unexpectedly: {stderr or '(no stderr output)'}"
        else:
            status, error_message = self._verify_output(session)

        result = self._write_result(session, status, error_message)

        final_state = {
            RecordingStatus.SUCCESS: RecorderState.COMPLETED,
            RecordingStatus.CANCELLED: RecorderState.CANCELLED,
            RecordingStatus.FAILED: RecorderState.FAILED,
            RecordingStatus.DEVICE_BUSY: RecorderState.FAILED,
        }[status]
        with self._lock:
            self._set_state(final_state, reason)
        session.finalized_event.set()

        return result

    def _verify_output(self, session: _Session):
        """Confirm the output file exists, is non-empty, and is a plausible size."""
        path = session.output_path
        if not path.exists():
            return RecordingStatus.FAILED, f"Output file was not created: {path}"
        size = path.stat().st_size
        if size <= 0:
            return RecordingStatus.FAILED, f"Output file is empty: {path}"

        if not self.config.simulate and session.actual_start and session.actual_stop:
            elapsed = (session.actual_stop - session.actual_start).total_seconds()
            expected = expected_iq_file_size(session.sample_rate, elapsed)
            if expected > 0:
                ratio = size / expected
                logger.info(
                    "File size check for %s: expected ~%d bytes, actual %d bytes (%.1f%%)",
                    path.name, expected, size, ratio * 100,
                )
                if ratio < (1 - self.config.file_size_tolerance):
                    return RecordingStatus.FAILED, (
                        f"Output file smaller than expected: {size} bytes vs ~{expected} expected "
                        f"({ratio * 100:.1f}%) - the recording may have failed partway through"
                    )
        return RecordingStatus.SUCCESS, None

    def _write_result(self, session: _Session, status: RecordingStatus, error_message: Optional[str]) -> RecordingResult:
        output_path = session.output_path
        size = output_path.stat().st_size if output_path.exists() else None
        duration = None
        if session.actual_start and session.actual_stop:
            duration = (session.actual_stop - session.actual_start).total_seconds()
        expected_size = expected_iq_file_size(session.sample_rate, duration) if duration else None

        metadata = RecordingMetadata(
            satellite_name=session.satellite_name,
            norad_id=session.norad_id,
            frequency_hz=session.frequency_hz,
            sample_rate=session.sample_rate,
            gain=session.gain,
            output_filename=output_path.name,
            recording_status=status.value,
            scheduled_aos=to_iso(session.scheduled_aos),
            scheduled_los=to_iso(session.scheduled_los),
            actual_recording_start=to_iso(session.actual_start),
            actual_recording_stop=to_iso(session.actual_stop),
            pre_buffer_seconds=session.pre_buffer_seconds,
            post_buffer_seconds=session.post_buffer_seconds,
            recording_duration_seconds=duration,
            output_file_size=size,
            expected_file_size=expected_size,
            error_message=error_message,
            simulated=self.config.simulate,
            device_index=self.config.device_index,
            command=session.command,
        )
        try:
            metadata.write(session.metadata_path)
            logger.info("Metadata written: %s", session.metadata_path)
        except OSError as exc:
            logger.error("Failed to write metadata file %s: %s", session.metadata_path, exc)

        result = RecordingResult(
            status=status,
            output_file=str(output_path) if output_path.exists() else None,
            metadata_file=str(session.metadata_path) if session.metadata_path.exists() else None,
            error_message=error_message,
            actual_recording_start=to_iso(session.actual_start),
            actual_recording_stop=to_iso(session.actual_stop),
            recording_duration_seconds=duration,
            output_file_size=size,
            metadata=metadata,
        )
        session.result = result

        if status == RecordingStatus.SUCCESS:
            logger.info("Recording completed: %s (%d bytes)", output_path.name, size or 0)
        elif status == RecordingStatus.CANCELLED:
            logger.warning("Recording cancelled: %s", output_path.name if output_path.exists() else "(no file)")
        else:
            logger.error("Recording failed: %s", error_message)
        return result
