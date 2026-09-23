"""Tests for rtl_recorder.scheduler: wall-clock wait-until-a-time-of-day logic."""

import threading
from datetime import datetime, time as dt_time, timedelta

from rtl_recorder.scheduler import next_occurrence, wait_until


class TestNextOccurrence:
    def test_target_later_today_returns_today(self):
        now = datetime(2026, 8, 24, 15, 0, 0)
        result = next_occurrence(dt_time(19, 0), now=now)
        assert result == datetime(2026, 8, 24, 19, 0, 0)

    def test_target_already_passed_today_returns_tomorrow(self):
        now = datetime(2026, 8, 24, 20, 0, 0)
        result = next_occurrence(dt_time(19, 0), now=now)
        assert result == datetime(2026, 8, 25, 19, 0, 0)

    def test_target_exactly_now_returns_tomorrow(self):
        """Exact match is treated as already-passed, not 'wait zero seconds'."""
        now = datetime(2026, 8, 24, 19, 0, 0)
        result = next_occurrence(dt_time(19, 0), now=now)
        assert result == datetime(2026, 8, 25, 19, 0, 0)

    def test_defaults_to_current_time_when_now_omitted(self):
        before = datetime.now()
        result = next_occurrence(dt_time(23, 59, 59, 999999))
        # Whatever "now" turned out to be, the result must be in the future
        # (never in the past) - the only real guarantee without injecting `now`.
        assert result > before

    def test_preserves_seconds_and_microseconds(self):
        now = datetime(2026, 8, 24, 10, 0, 0)
        result = next_occurrence(dt_time(19, 30, 15, 500), now=now)
        assert result == datetime(2026, 8, 24, 19, 30, 15, 500)


class TestWaitUntil:
    def test_returns_true_immediately_if_target_already_passed(self):
        now = datetime(2026, 8, 24, 19, 0, 0)
        target = now - timedelta(seconds=5)
        reached = wait_until(target, now_fn=lambda: now, sleep_fn=lambda s: None)
        assert reached is True

    def test_advances_simulated_clock_until_target_reached(self):
        state = {"now": datetime(2026, 8, 24, 18, 59, 55)}
        target = datetime(2026, 8, 24, 19, 0, 0)

        def fake_now():
            return state["now"]

        def fake_sleep(seconds):
            state["now"] += timedelta(seconds=seconds)

        reached = wait_until(target, poll_interval=1.0, now_fn=fake_now, sleep_fn=fake_sleep)

        assert reached is True
        assert state["now"] >= target

    def test_sleep_never_overshoots_target_by_more_than_poll_interval(self):
        state = {"now": datetime(2026, 8, 24, 18, 59, 0)}
        target = datetime(2026, 8, 24, 19, 0, 0)
        sleep_calls = []

        def fake_now():
            return state["now"]

        def fake_sleep(seconds):
            sleep_calls.append(seconds)
            state["now"] += timedelta(seconds=seconds)

        wait_until(target, poll_interval=10.0, now_fn=fake_now, sleep_fn=fake_sleep)

        assert all(s <= 10.0 for s in sleep_calls)
        assert state["now"] == target  # final sleep is clamped to exactly the remaining time

    def test_cancel_event_stops_the_wait_early(self):
        state = {"now": datetime(2026, 8, 24, 18, 0, 0)}
        target = datetime(2026, 8, 24, 19, 0, 0)  # an hour away
        cancel_event = threading.Event()
        call_count = {"n": 0}

        def fake_now():
            return state["now"]

        def fake_sleep(seconds):
            call_count["n"] += 1
            state["now"] += timedelta(seconds=seconds)
            if call_count["n"] == 3:
                cancel_event.set()

        reached = wait_until(target, poll_interval=60.0, cancel_event=cancel_event, now_fn=fake_now, sleep_fn=fake_sleep)

        assert reached is False
        assert state["now"] < target

    def test_no_cancel_event_never_cancels(self):
        state = {"now": datetime(2026, 8, 24, 18, 59, 58)}
        target = datetime(2026, 8, 24, 19, 0, 0)

        def fake_sleep(seconds):
            state["now"] += timedelta(seconds=seconds)

        reached = wait_until(target, poll_interval=1.0, cancel_event=None, now_fn=lambda: state["now"], sleep_fn=fake_sleep)
        assert reached is True
