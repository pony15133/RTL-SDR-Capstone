"""Wall-clock scheduling: wait until a specific time of day (or an exact
date+time), then let the caller start a recording.

This is deliberately independent of satellite pass prediction (AOS/LOS,
pre/post buffers) - that is ``RTLSDRRecorder.record_pass()``, still
unimplemented (Phase 2). This module only answers a much simpler
question: "it's not 7pm yet - wait until it is, without blocking forever
and without ignoring a cancel request." Manual/FM-station-style
"auto-capture at a fixed time" scheduling belongs here; satellite-pass
scheduling belongs in record_pass().
"""

import logging
import threading
import time as time_module
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def next_occurrence(target_time: dt_time, now: Optional[datetime] = None) -> datetime:
    """The next datetime (today or tomorrow) at which the clock reads target_time.

    If ``now``'s time-of-day is already at or past ``target_time``, returns
    tomorrow's occurrence; otherwise returns today's. ``now`` defaults to
    the current local time. The result carries whatever tzinfo ``now``
    has (naive in, naive out; aware in, aware out) - callers should be
    consistent about naive-vs-aware rather than mixing them, the same way
    the rest of this codebase expects for datetimes.
    """
    if now is None:
        now = datetime.now()
    candidate = now.replace(
        hour=target_time.hour, minute=target_time.minute,
        second=target_time.second, microsecond=target_time.microsecond,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def wait_until(
    target_dt: datetime,
    *,
    cancel_event: Optional[threading.Event] = None,
    poll_interval: float = 1.0,
    now_fn: Callable[[], datetime] = datetime.now,
    sleep_fn: Callable[[float], None] = time_module.sleep,
) -> bool:
    """Block until target_dt, or until cancel_event is set. Returns True if
    target_dt was reached, False if cancelled early.

    Sleeps in short increments (poll_interval) rather than one long sleep,
    so a cancel request or a KeyboardInterrupt is noticed promptly instead
    of only after the full wait. ``now_fn``/``sleep_fn`` are injectable so
    this is deterministically testable without a real wait.
    """
    while True:
        now = now_fn()
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return True
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Scheduled wait cancelled with %.1fs remaining", remaining)
            return False
        sleep_fn(min(poll_interval, remaining))
