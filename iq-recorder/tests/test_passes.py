"""Pass prediction (rtl_recorder/passes.py) and scheduled recording
(RTLSDRRecorder.record_pass)."""

import math
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from rtl_recorder import RTLSDRRecorder, RecorderConfig
from rtl_recorder.passes import (
    TLE,
    GroundStation,
    KeplerJ2Propagator,
    doppler_shift_hz,
    eci_to_ecef,
    elevation_at,
    find_passes,
    get_tle,
    gmst_rad,
    look_angles,
    make_propagator,
    parse_tles,
)
from rtl_recorder.states import RecordingStatus


def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


# Vallado et al., "Revisiting Spacetrack Report #3" verification case 00005.
VANGUARD = TLE(
    "VANGUARD 1",
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
)
# Official SGP4 output (TEME, km) at t = 0 and t = 360 min.
VANGUARD_REF = {0: (7022.46529266, -1400.08296755, 0.03995155),
                360: (-7154.03120202, -3783.17682504, -3536.19412294)}

ISS_LIKE = TLE(
    "ISS (ZARYA)",
    _fix_checksum("1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"),
    _fix_checksum("2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"),
)
SINGAPORE = GroundStation(1.3521, 103.8198, 15, "Singapore")


class TestTLE:
    def test_fields(self):
        t = VANGUARD
        assert t.norad_id == 5
        assert t.epoch == datetime(2000, 6, 27, 18, 50, 19, 733568, tzinfo=timezone.utc)
        assert t.eccentricity == pytest.approx(0.1859667)
        assert t.inclination_deg == pytest.approx(34.2682)
        assert t.bstar == pytest.approx(2.8098e-5)
        assert t.period_minutes == pytest.approx(1440 / 10.82419157)

    def test_parse_two_and_three_line_sets(self):
        text = f"0 ISS (ZARYA)\n{ISS_LIKE.line1}\n{ISS_LIKE.line2}\n{VANGUARD.line1}\n{VANGUARD.line2}\n"
        tles = parse_tles(text)
        assert [t.norad_id for t in tles] == [25544, 5]
        assert tles[0].name == "ISS (ZARYA)"

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            TLE("x", "not a tle", "at all")


def _geodetic_subpoint(r_ecef):
    """ECEF -> geodetic (lat, lon) of the point directly below (WGS84, iterative)."""
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    x, y, z = r_ecef
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - e2 * n / (n + h)))
    return math.degrees(lat), math.degrees(math.atan2(y, x))


class TestFrames:
    def test_gmst_at_j2000(self):
        assert math.degrees(gmst_rad(datetime(2000, 1, 1, 12, tzinfo=timezone.utc))) == pytest.approx(280.46061837, abs=1e-6)

    def test_station_directly_below_sees_zenith(self):
        when = ISS_LIKE.epoch
        r = eci_to_ecef(KeplerJ2Propagator(ISS_LIKE).position_eci(when), when)
        lat, lon = _geodetic_subpoint(r)
        _, el, rng = look_angles(r, GroundStation(lat, lon))
        assert el > 89.9
        assert 300 < rng < 450  # roughly the altitude of an ISS-like orbit

    def test_opposite_side_of_earth_is_below_horizon(self):
        when = ISS_LIKE.epoch
        r = eci_to_ecef(KeplerJ2Propagator(ISS_LIKE).position_eci(when), when)
        lat = math.degrees(math.atan2(r[2], math.hypot(r[0], r[1])))
        lon = math.degrees(math.atan2(r[1], r[0]))
        assert look_angles(r, GroundStation(-lat, lon + 180))[1] < -60


class TestPropagation:
    @pytest.mark.parametrize("minutes", [0, 360])
    def test_fallback_matches_official_sgp4_within_tens_of_km(self, minutes):
        r = KeplerJ2Propagator(VANGUARD).position_eci(VANGUARD.epoch + timedelta(minutes=minutes))
        assert math.dist(r, VANGUARD_REF[minutes]) < 20.0

    @pytest.mark.parametrize("minutes", [0, 360])
    def test_sgp4_backend_matches_reference(self, minutes):
        pytest.importorskip("sgp4")
        r = make_propagator(VANGUARD, "sgp4").position_eci(VANGUARD.epoch + timedelta(minutes=minutes))
        assert math.dist(r, VANGUARD_REF[minutes]) < 0.01

    def test_auto_backend_never_fails(self):
        assert make_propagator(ISS_LIKE).name in ("sgp4", "kepler-j2")


class TestFindPasses:
    @pytest.fixture(scope="class")
    def passes(self):
        return find_passes(ISS_LIKE, SINGAPORE, start=ISS_LIKE.epoch, hours=24, min_max_elevation_deg=10)

    def test_finds_a_few_leo_passes_per_day(self, passes):
        assert 2 <= len(passes) <= 8

    def test_pass_geometry_is_consistent(self, passes):
        prop = make_propagator(ISS_LIKE)
        for p in passes:
            assert p.aos < p.max_elevation_time < p.los
            assert 3 * 60 < p.duration_seconds < 15 * 60      # LEO passes last minutes
            assert p.max_elevation_deg >= 10
            assert elevation_at(prop, SINGAPORE, p.aos)[0] == pytest.approx(0, abs=0.3)
            assert elevation_at(prop, SINGAPORE, p.los)[0] == pytest.approx(0, abs=0.3)
            assert elevation_at(prop, SINGAPORE, p.max_elevation_time)[0] == pytest.approx(p.max_elevation_deg, abs=0.1)

    def test_doppler_sign_and_size(self, passes):
        prop = make_propagator(ISS_LIKE)
        p = passes[0]
        f = 437e6
        rising = doppler_shift_hz(prop, SINGAPORE, f, p.aos + timedelta(seconds=10))
        setting = doppler_shift_hz(prop, SINGAPORE, f, p.los - timedelta(seconds=10))
        assert rising > 0 > setting                      # approaching -> higher frequency
        assert abs(rising) < 7.8 / 299792.458 * f * 1.05  # can't exceed orbital speed

    def test_min_elevation_filter(self):
        all_passes = find_passes(ISS_LIKE, SINGAPORE, start=ISS_LIKE.epoch, hours=24, min_max_elevation_deg=0)
        high = find_passes(ISS_LIKE, SINGAPORE, start=ISS_LIKE.epoch, hours=24, min_max_elevation_deg=40)
        assert len(high) < len(all_passes)
        assert all(p.max_elevation_deg >= 40 for p in high)

    def test_pass_in_progress_starts_at_search_start(self, passes):
        p = passes[0]
        mid = p.aos + timedelta(seconds=60)
        again = find_passes(ISS_LIKE, SINGAPORE, start=mid, hours=1, min_max_elevation_deg=0)
        assert again[0].aos == mid
        assert abs((again[0].los - p.los).total_seconds()) < 2


class TestGetTLE:
    def test_downloads_then_uses_cache(self, tmp_path):
        calls = []

        def fake_fetch(url):
            calls.append(url)
            return f"ISS (ZARYA)\n{ISS_LIKE.line1}\n{ISS_LIKE.line2}\n"

        t1 = get_tle(25544, cache_dir=tmp_path, fetch=fake_fetch)
        t2 = get_tle(25544, cache_dir=tmp_path, fetch=fake_fetch)
        assert t1.norad_id == t2.norad_id == 25544
        assert len(calls) == 1 and "CATNR=25544" in calls[0]

    def test_stale_cache_used_when_offline(self, tmp_path):
        (tmp_path / "25544.tle").write_text(f"ISS\n{ISS_LIKE.line1}\n{ISS_LIKE.line2}\n")

        def offline(url):
            raise OSError("no network")

        future = datetime.now(timezone.utc) + timedelta(days=3)  # cache is "stale"
        assert get_tle(25544, cache_dir=tmp_path, fetch=offline, now=future).norad_id == 25544

    def test_no_cache_and_offline_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not get a TLE"):
            get_tle(25544, cache_dir=tmp_path, fetch=lambda url: (_ for _ in ()).throw(OSError("down")))


class TestRecordPass:
    def _recorder(self, tmp_path):
        return RTLSDRRecorder(RecorderConfig(simulate=True, output_dir=str(tmp_path), monitor_interval_seconds=0.05))

    def _kwargs(self, **over):
        kw = dict(satellite_name="TEST-SAT", frequency_hz=437_000_000, sample_rate=2_400_000, gain=30, norad_id=25544)
        kw.update(over)
        return kw

    def test_waits_for_aos_then_records_with_buffers(self, tmp_path):
        fake_now = [datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)]
        real_start = time.monotonic()

        def now_fn():
            return fake_now[0] + timedelta(seconds=time.monotonic() - real_start)

        def sleep_fn(seconds):
            fake_now[0] += timedelta(seconds=seconds)  # jump the clock instead of sleeping

        aos = fake_now[0] + timedelta(hours=2)
        result = self._recorder(tmp_path).record_pass(
            **self._kwargs(), aos=aos, los=aos + timedelta(seconds=1), pre_buffer=0.5, post_buffer=0.5,
            now_fn=now_fn, sleep_fn=sleep_fn,
        )
        assert result.status == RecordingStatus.SUCCESS
        assert result.metadata.scheduled_aos == aos.isoformat()
        assert result.metadata.pre_buffer_seconds == 0.5
        assert result.recording_duration_seconds == pytest.approx(2.0, abs=0.6)

    def test_joins_a_pass_already_in_progress(self, tmp_path):
        now = datetime.now(timezone.utc)
        result = self._recorder(tmp_path).record_pass(
            **self._kwargs(), aos=now - timedelta(minutes=5), los=now + timedelta(seconds=1),
            pre_buffer=0, post_buffer=0.5,
        )
        assert result.status == RecordingStatus.SUCCESS
        assert result.recording_duration_seconds < 3  # only the remainder, not 5 minutes

    def test_pass_already_over_is_refused(self, tmp_path):
        now = datetime.now(timezone.utc)
        result = self._recorder(tmp_path).record_pass(
            **self._kwargs(), aos=now - timedelta(minutes=20), los=now - timedelta(minutes=10))
        assert result.status == RecordingStatus.FAILED
        assert "already over" in result.error_message

    def test_los_before_aos_is_refused(self, tmp_path):
        now = datetime.now(timezone.utc)
        result = self._recorder(tmp_path).record_pass(**self._kwargs(), aos=now, los=now - timedelta(seconds=1))
        assert result.status == RecordingStatus.FAILED

    def test_cancel_while_waiting(self, tmp_path):
        rec = self._recorder(tmp_path)
        now = datetime.now(timezone.utc)
        box = {}
        worker = threading.Thread(target=lambda: box.update(r=rec.record_pass(
            **self._kwargs(), aos=now + timedelta(hours=1), los=now + timedelta(hours=1, minutes=10))))
        worker.start()
        for _ in range(100):
            if rec.check_recording_status().value == "WAITING":
                break
            time.sleep(0.02)
        cancel = rec.cancel_recording()
        worker.join(timeout=5)
        assert cancel.status == RecordingStatus.CANCELLED
        assert box["r"].status == RecordingStatus.CANCELLED
