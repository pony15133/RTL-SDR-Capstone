"""Satellite pass prediction from TLE orbital elements.

This is what turns "record METEOR-M2-4" into "start at 14:03:12, stop at
14:15:40": fetch (or read) the satellite's TLE, propagate its orbit, and
find when it rises above / sets below the horizon for the ground station.

    from rtl_recorder.passes import GroundStation, get_tle, find_passes
    station = GroundStation(lat_deg=1.3521, lon_deg=103.8198, alt_m=15)   # Singapore
    tle = get_tle(40069)                                                  # METEOR-M2-4? use your id
    for p in find_passes(tle, station, hours=24):
        print(p.aos, p.los, round(p.max_elevation_deg))

Propagation backends:

* ``sgp4`` (pip install sgp4) - the standard SGP4 model the TLEs are built
  for; used automatically when installed. Recommended.
* built-in fallback - two-body Kepler orbit plus the dominant J2 secular
  drift (node regression, apsidal rotation) and the TLE's drag term. Good
  to roughly tens of km for a fresh TLE, i.e. pass times within about
  +/-10-30 s, which the recording pre/post buffers absorb. It exists so the
  scheduler still works on a machine without sgp4.

TLE sources: CelesTrak's GP API (cached on disk, refreshed every 12 h, the
cache is used if the network is down) or a local TLE file.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MU_EARTH = 398600.4418          # km^3/s^2
R_EARTH = 6378.137              # km (WGS84 equatorial)
J2 = 1.08262668e-3
WGS84_F = 1 / 298.257223563
C_KM_S = 299792.458
SIDEREAL_DAY_S = 86164.0905

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=tle"
DEFAULT_CACHE_DIR = Path("tle_cache")
DEFAULT_MAX_TLE_AGE_HOURS = 12.0


class TLEError(ValueError):
    """The TLE text could not be parsed."""


# --------------------------------------------------------------------------- #
# TLE parsing / fetching
# --------------------------------------------------------------------------- #

def _checksum_ok(line: str) -> bool:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return len(line) >= 69 and line[68].isdigit() and total % 10 == int(line[68])


def _implied_decimal(field: str) -> float:
    """TLE 'assumed decimal point' fields like ' 28098-4' -> 0.28098e-4."""
    field = field.strip()
    if not field or field in ("00000-0", "00000+0", "0"):
        return 0.0
    sign = -1.0 if field[0] == "-" else 1.0
    field = field.lstrip("+-")
    mantissa, exp = field[:-2], field[-2:]
    return sign * float(f"0.{mantissa}") * 10 ** int(exp)


@dataclass
class TLE:
    name: str
    line1: str
    line2: str

    def __post_init__(self):
        l1, l2 = self.line1.rstrip(), self.line2.rstrip()
        if not (l1.startswith("1 ") and l2.startswith("2 ")) or len(l1) < 64 or len(l2) < 63:
            raise TLEError(f"Not a valid TLE pair for {self.name!r}")
        for line in (l1, l2):
            if not _checksum_ok(line):
                logger.warning("TLE checksum mismatch for %s - continuing, but check the source", self.name)
        self.line1, self.line2 = l1, l2
        self.norad_id = int(l1[2:7])
        year = int(l1[18:20])
        year += 2000 if year < 57 else 1900
        day = float(l1[20:32])
        self.epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
        self.ndot_over_2 = float(l1[33:43])          # rev/day^2
        self.bstar = _implied_decimal(l1[53:61])
        self.inclination_deg = float(l2[8:16])
        self.raan_deg = float(l2[17:25])
        self.eccentricity = float("0." + l2[26:33].strip())
        self.arg_perigee_deg = float(l2[34:42])
        self.mean_anomaly_deg = float(l2[43:51])
        self.mean_motion_rev_day = float(l2[52:63])

    @property
    def period_minutes(self) -> float:
        return 1440.0 / self.mean_motion_rev_day

    def age_days(self, when: Optional[datetime] = None) -> float:
        when = when or datetime.now(timezone.utc)
        return (when - self.epoch).total_seconds() / 86400.0


def parse_tles(text: str) -> List[TLE]:
    """Parse 2-line or 3-line (name + 2 lines) TLE sets from text."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: List[TLE] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            out.append(TLE(f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i].strip()
            if name.startswith("0 "):  # 3LE format prefixes the name with "0 "
                name = name[2:].strip()
            out.append(TLE(name, lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def load_tle_file(path, norad_id: Optional[int] = None) -> TLE:
    tles = parse_tles(Path(path).read_text(encoding="utf-8"))
    if not tles:
        raise TLEError(f"No TLEs found in {path}")
    if norad_id is None:
        return tles[0]
    for tle in tles:
        if tle.norad_id == int(norad_id):
            return tle
    raise TLEError(f"NORAD {norad_id} not in {path}")


def get_tle(norad_id: int, *, cache_dir: Path = DEFAULT_CACHE_DIR, max_age_hours: float = DEFAULT_MAX_TLE_AGE_HOURS,
            fetch: Optional[Callable[[str], str]] = None, now: Optional[datetime] = None) -> TLE:
    """Latest TLE for a NORAD id: fresh cache -> CelesTrak -> stale cache.

    ``fetch`` (url -> text) is injectable for tests/offline use.
    """
    cache_dir = Path(cache_dir)
    cache = cache_dir / f"{int(norad_id)}.tle"
    now = now or datetime.now(timezone.utc)
    if cache.exists():
        age_h = (now.timestamp() - cache.stat().st_mtime) / 3600.0
        if age_h <= max_age_hours:
            return load_tle_file(cache, norad_id)
    url = CELESTRAK_URL.format(norad=int(norad_id))
    try:
        text = (fetch or _http_get)(url)
        tles = [t for t in parse_tles(text) if t.norad_id == int(norad_id)]
        if not tles:
            raise TLEError(f"CelesTrak returned no TLE for NORAD {norad_id}: {text[:80]!r}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(f"{tles[0].name}\n{tles[0].line1}\n{tles[0].line2}\n", encoding="utf-8")
        return tles[0]
    except Exception as exc:
        if cache.exists():
            logger.warning("TLE download failed (%s) - using cached TLE %s", exc, cache)
            return load_tle_file(cache, norad_id)
        raise TLEError(f"Could not get a TLE for NORAD {norad_id}: {exc}") from exc


def _http_get(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "rtl-sdr-capstone/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return resp.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Time / frames
# --------------------------------------------------------------------------- #

def julian_date(when: datetime) -> float:
    when = when.astimezone(timezone.utc)
    return when.timestamp() / 86400.0 + 2440587.5


def gmst_rad(when: datetime) -> float:
    """Greenwich mean sidereal time (IAU 1982), radians."""
    jd = julian_date(when)
    t = (jd - 2451545.0) / 36525.0
    seconds = (67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t ** 3)
    return math.radians((seconds % 86400.0) / 240.0) % (2 * math.pi)


def eci_to_ecef(r: Sequence[float], when: datetime) -> Tuple[float, float, float]:
    g = gmst_rad(when)
    c, s = math.cos(g), math.sin(g)
    return (c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2])


@dataclass(frozen=True)
class GroundStation:
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    name: str = "station"

    def ecef_km(self) -> Tuple[float, float, float]:
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        e2 = WGS84_F * (2 - WGS84_F)
        n = R_EARTH / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = self.alt_m / 1000.0
        return ((n + h) * math.cos(lat) * math.cos(lon),
                (n + h) * math.cos(lat) * math.sin(lon),
                (n * (1 - e2) + h) * math.sin(lat))


def look_angles(sat_ecef: Sequence[float], station: GroundStation) -> Tuple[float, float, float]:
    """(azimuth deg, elevation deg, range km) of the satellite from the station."""
    ox, oy, oz = station.ecef_km()
    dx, dy, dz = sat_ecef[0] - ox, sat_ecef[1] - oy, sat_ecef[2] - oz
    lat, lon = math.radians(station.lat_deg), math.radians(station.lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    east = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    up = cl * co * dx + cl * so * dy + sl * dz
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    if rng == 0.0:
        return 0.0, 90.0, 0.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, up / rng))))  # clamp: float error at exact zenith
    az = math.degrees(math.atan2(east, north)) % 360.0
    return az, el, rng


# --------------------------------------------------------------------------- #
# Propagators
# --------------------------------------------------------------------------- #

class KeplerJ2Propagator:
    """Two-body orbit + J2 secular rates + TLE drag term (fallback model)."""

    name = "kepler-j2"

    def __init__(self, tle: TLE):
        self.tle = tle
        self.n0 = tle.mean_motion_rev_day * 2 * math.pi / 86400.0            # rad/s
        self.a = (MU_EARTH / self.n0 ** 2) ** (1.0 / 3.0)
        e, i = tle.eccentricity, math.radians(tle.inclination_deg)
        p = self.a * (1 - e * e)
        k = 1.5 * J2 * (R_EARTH / p) ** 2 * self.n0
        self.raan_dot = -k * math.cos(i)
        self.argp_dot = 0.5 * k * (5 * math.cos(i) ** 2 - 1)
        self.e, self.i = e, i

    def position_eci(self, when: datetime) -> Tuple[float, float, float]:
        t = self.tle
        dt = (when - t.epoch).total_seconds()
        dt_days = dt / 86400.0
        mean_anom = math.radians(t.mean_anomaly_deg) + self.n0 * dt + 2 * math.pi * t.ndot_over_2 * dt_days ** 2
        n_now = self.n0 + 4 * math.pi * t.ndot_over_2 * dt_days / 86400.0
        a = (MU_EARTH / n_now ** 2) ** (1.0 / 3.0)
        raan = math.radians(t.raan_deg) + self.raan_dot * dt
        argp = math.radians(t.arg_perigee_deg) + self.argp_dot * dt
        e = self.e
        E = mean_anom % (2 * math.pi)
        for _ in range(30):  # Newton on Kepler's equation
            dE = (E - e * math.sin(E) - mean_anom % (2 * math.pi)) / (1 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-12:
                break
        x_p = a * (math.cos(E) - e)
        y_p = a * math.sqrt(1 - e * e) * math.sin(E)
        co, so = math.cos(raan), math.sin(raan)
        cw, sw = math.cos(argp), math.sin(argp)
        ci, si = math.cos(self.i), math.sin(self.i)
        return (
            (co * cw - so * sw * ci) * x_p + (-co * sw - so * cw * ci) * y_p,
            (so * cw + co * sw * ci) * x_p + (-so * sw + co * cw * ci) * y_p,
            (sw * si) * x_p + (cw * si) * y_p,
        )


class SGP4Propagator:
    """The standard SGP4 model via the ``sgp4`` package."""

    name = "sgp4"

    def __init__(self, tle: TLE):
        from sgp4.api import Satrec

        self.tle = tle
        self.sat = Satrec.twoline2rv(tle.line1, tle.line2)

    def position_eci(self, when: datetime) -> Tuple[float, float, float]:
        jd = julian_date(when)
        whole = math.floor(jd)
        err, r, _ = self.sat.sgp4(whole, jd - whole)
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err} for NORAD {self.tle.norad_id} at {when.isoformat()}")
        return tuple(r)


def make_propagator(tle: TLE, backend: str = "auto"):
    if backend in ("auto", "sgp4"):
        try:
            return SGP4Propagator(tle)
        except ImportError:
            if backend == "sgp4":
                raise
            logger.info("sgp4 package not installed - using the built-in Kepler+J2 propagator "
                        "(pip install sgp4 for full accuracy)")
    return KeplerJ2Propagator(tle)


# --------------------------------------------------------------------------- #
# Pass search
# --------------------------------------------------------------------------- #

@dataclass
class Pass:
    satellite_name: str
    norad_id: int
    aos: datetime
    los: datetime
    max_elevation_deg: float
    max_elevation_time: datetime
    aos_azimuth_deg: float
    los_azimuth_deg: float
    propagator: str

    @property
    def duration_seconds(self) -> float:
        return (self.los - self.aos).total_seconds()

    def doppler_hz(self, propagator, station: GroundStation, frequency_hz: float, when: datetime) -> float:
        """Received frequency offset at ``when`` (positive while approaching)."""
        return doppler_shift_hz(propagator, station, frequency_hz, when)

    def describe(self) -> str:
        return (f"{self.satellite_name} (NORAD {self.norad_id}): AOS {self.aos:%Y-%m-%d %H:%M:%S}Z "
                f"az {self.aos_azimuth_deg:5.1f} -> max el {self.max_elevation_deg:4.1f} at "
                f"{self.max_elevation_time:%H:%M:%S}Z -> LOS {self.los:%H:%M:%S}Z az {self.los_azimuth_deg:5.1f} "
                f"({self.duration_seconds / 60:.1f} min)")


def elevation_at(propagator, station: GroundStation, when: datetime) -> Tuple[float, float]:
    az, el, _ = look_angles(eci_to_ecef(propagator.position_eci(when), when), station)
    return el, az


def doppler_shift_hz(propagator, station: GroundStation, frequency_hz: float, when: datetime) -> float:
    h = timedelta(seconds=0.5)
    _, _, r1 = look_angles(eci_to_ecef(propagator.position_eci(when - h), when - h), station)
    _, _, r2 = look_angles(eci_to_ecef(propagator.position_eci(when + h), when + h), station)
    range_rate = (r2 - r1) / 1.0  # km/s
    return -range_rate / C_KM_S * frequency_hz


def _refine_crossing(propagator, station, t_lo, t_hi, horizon_deg, rising: bool) -> datetime:
    """Bisect to ~0.5 s the moment elevation crosses horizon_deg between t_lo and t_hi."""
    for _ in range(40):
        if (t_hi - t_lo).total_seconds() <= 0.5:
            break
        mid = t_lo + (t_hi - t_lo) / 2
        above = elevation_at(propagator, station, mid)[0] >= horizon_deg
        if above == rising:
            t_hi = mid
        else:
            t_lo = mid
    return t_hi if rising else t_lo


def find_passes(tle: TLE, station: GroundStation, *, start: Optional[datetime] = None, hours: float = 24.0,
                horizon_deg: float = 0.0, min_max_elevation_deg: float = 10.0, step_seconds: float = 20.0,
                backend: str = "auto") -> List[Pass]:
    """All passes that start within [start, start+hours] and peak above
    ``min_max_elevation_deg``. AOS/LOS are when elevation crosses ``horizon_deg``.
    A pass already in progress at ``start`` is included with AOS = start."""
    propagator = make_propagator(tle, backend)
    start = (start or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end = start + timedelta(hours=hours)
    step = timedelta(seconds=step_seconds)

    passes: List[Pass] = []
    t = start
    el_prev, _ = elevation_at(propagator, station, t)
    aos = start if el_prev >= horizon_deg else None
    best = (el_prev, t) if aos else (-90.0, t)
    while True:
        t_next = t + step
        el, _ = elevation_at(propagator, station, t_next)
        if aos is None and el >= horizon_deg:
            if t_next > end:
                break
            aos = _refine_crossing(propagator, station, t, t_next, horizon_deg, rising=True)
            best = (el, t_next)
        elif aos is not None:
            if el > best[0]:
                best = (el, t_next)
            if el < horizon_deg:
                los = _refine_crossing(propagator, station, t, t_next, horizon_deg, rising=False)
                max_el, max_t = _refine_max(propagator, station, best[1], step)
                if max_el >= min_max_elevation_deg:
                    passes.append(Pass(
                        satellite_name=tle.name, norad_id=tle.norad_id, aos=aos, los=los,
                        max_elevation_deg=max_el, max_elevation_time=max_t,
                        aos_azimuth_deg=elevation_at(propagator, station, aos)[1],
                        los_azimuth_deg=elevation_at(propagator, station, los)[1],
                        propagator=propagator.name,
                    ))
                aos = None
                best = (-90.0, t_next)
        t = t_next
        if aos is None and t > end:
            break
        if t > end + timedelta(hours=2):  # a pass can't last 2 h for LEO - safety stop
            break
    return passes


def _refine_max(propagator, station, around: datetime, step: timedelta) -> Tuple[float, datetime]:
    lo, hi = around - step, around + step
    for _ in range(40):  # golden-section search on elevation
        if (hi - lo).total_seconds() < 0.5:
            break
        m1 = lo + (hi - lo) * 0.382
        m2 = lo + (hi - lo) * 0.618
        if elevation_at(propagator, station, m1)[0] < elevation_at(propagator, station, m2)[0]:
            lo = m1
        else:
            hi = m2
    mid = lo + (hi - lo) / 2
    return elevation_at(propagator, station, mid)[0], mid
