#!/usr/bin/env python3
"""
gnss_offline_path.py

A practical GNSS offline path solver for:
- RINEX 4.x observation files (GPS-only, pseudorange C1C/C1W/C1P/C1S/C1L/C1X)
- SP3 precise orbit files (Lagrange polynomial interpolation)
- IGS precise clock files (.clk) for sub-metre satellite clock accuracy  [NEW]
- RINEX NAV files for Klobuchar ionospheric correction coefficients       [NEW]
- 1 Hz trajectory output to CSV + KML

Accuracy improvements over v1:
- Saastamoinen tropospheric delay correction per satellite                [NEW]
- Klobuchar single-frequency ionospheric correction (needs --nav)        [NEW]
- Precise satellite clocks from IGS CLK files (needs --clk)              [NEW]
  SP3 clocks are used as fallback when a CLK file is not provided or
  when a satellite is missing from the CLK file.
- SP3 clock sentinel-value filtering (|clk| > 0.999 s discarded)        [NEW]

All improvements are additive and backward-compatible.

Automatic download behavior:
- If --sp3 is omitted, the script uses gnss-lib-py to download the needed SP3.
- If --auto-clk is supplied and --clk is omitted, it also tries to download CLK.
- RINEX 4 parsing is still done by this script, not by gnss-lib-py.

Typical use with automatic SP3 download:
    python gnss_offline_path_auto_sp3.py \
        --obs gnss_log_2026_03_21_17_17_57.26o \
        --out-prefix results/track \
        --gps-utc-leap-seconds 18 \
        --elevation-mask-deg 10

Typical use with automatic SP3 + CLK download:
    python gnss_offline_path_auto_sp3.py \
        --obs gnss_log_2026_03_21_17_17_57.26o \
        --out-prefix results/track \
        --gps-utc-leap-seconds 18 \
        --elevation-mask-deg 10 \
        --auto-clk
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical / geodetic constants
# ---------------------------------------------------------------------------
C = 299_792_458.0          # speed of light, m/s
OMEGA_E = 7.2921151467e-5  # WGS-84 Earth rotation rate, rad/s
WGS84_A = 6378137.0        # semi-major axis, m
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
GPS_L1_FREQ = 1575.42e6    # Hz  (used for Klobuchar)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SatObs:
    sat_id: str
    pseudorange_m: float
    doppler_hz: Optional[float] = None
    snr: Optional[float] = None


@dataclass
class EpochObs:
    time_gps: dt.datetime
    sats: List[SatObs] = field(default_factory=list)


@dataclass
class Sp3Record:
    time: dt.datetime
    sat_positions_m: Dict[str, np.ndarray] = field(default_factory=dict)
    sat_clocks_s: Dict[str, float] = field(default_factory=dict)


@dataclass
class SolutionEpoch:
    time_gps: dt.datetime
    time_utc: dt.datetime
    x_m: float
    y_m: float
    z_m: float
    lat_deg: float
    lon_deg: float
    h_m: float
    clock_bias_m: float
    num_sats: int
    rms_residual_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    speed_mps: float = 0.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_float(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text.replace("D", "E"))
    except ValueError:
        return None


def gps_datetime_to_utc(gps_dt: dt.datetime, leap_seconds: int) -> dt.datetime:
    return gps_dt - dt.timedelta(seconds=leap_seconds)


def datetime_from_rinex_fields(year: int, month: int, day: int,
                               hour: int, minute: int, second_float: float) -> dt.datetime:
    sec_int = int(math.floor(second_float))
    micro = int(round((second_float - sec_int) * 1e6))
    if micro >= 1_000_000:
        sec_int += 1
        micro -= 1_000_000
    return dt.datetime(year, month, day, hour, minute, sec_int, micro)


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p < 1e-12:
        lat = math.copysign(math.pi / 2.0, z)
        h = abs(z) - WGS84_A * math.sqrt(1.0 - WGS84_E2)
        return math.degrees(lat), math.degrees(lon), h

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        cos_lat = math.cos(lat)
        if abs(cos_lat) < 1e-12:
            break
        h = p / cos_lat - n
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    cos_lat = math.cos(lat)
    h = (p / cos_lat - n) if abs(cos_lat) > 1e-12 else abs(z) / math.sqrt(1.0 - WGS84_E2) - n
    return math.degrees(lat), math.degrees(lon), h


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    return np.array([
        [-slon,        clon,       0.0],
        [-slat*clon,  -slat*slon,  clat],
        [ clat*clon,   clat*slon,  slat],
    ])


def elevation_angle_deg(rx_ecef: np.ndarray, sv_ecef: np.ndarray) -> float:
    lat, lon, _ = ecef_to_geodetic(*rx_ecef.tolist())
    enu_rot = ecef_to_enu_matrix(lat, lon)
    los = sv_ecef - rx_ecef
    los_unit = los / np.linalg.norm(los)
    enu = enu_rot @ los_unit
    up = enu[2]
    return math.degrees(math.asin(max(-1.0, min(1.0, up))))


def apply_earth_rotation_correction(sv_pos_ecef: np.ndarray, transit_time_s: float) -> np.ndarray:
    angle = OMEGA_E * transit_time_s
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([
        [cos_a, sin_a, 0.0],
        [-sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return rot @ sv_pos_ecef


# ---------------------------------------------------------------------------
# [NEW] Saastamoinen tropospheric correction
# ---------------------------------------------------------------------------

def saastamoinen_tropo_delay_m(h_m: float, elev_deg: float) -> float:
    """
    Compute the Saastamoinen tropospheric slant delay in metres.

    Uses a standard atmosphere model (no measured meteo data needed).
    Accuracy ~1 cm for elevations > 10°, degrades below that.

    Parameters
    ----------
    h_m      : receiver height above the WGS-84 ellipsoid, metres
    elev_deg : satellite elevation angle, degrees

    Returns
    -------
    Slant tropospheric delay in metres (always positive, added to pseudorange).
    """
    if elev_deg < 2.0:
        # Below 2° the mapping function diverges; clamp to avoid blowup.
        elev_deg = 2.0

    # Standard atmosphere at height h_m (ICAO lapse-rate model)
    P = 1013.25 * (1.0 - 2.2557e-5 * h_m) ** 5.2559   # pressure, hPa
    T = 15.0 - 6.5e-3 * h_m + 273.15                    # temperature, K
    # Partial pressure of water vapour: assume 50% relative humidity at 15 °C
    # saturated vapour pressure ~17.1 hPa at 15 °C
    e = 0.50 * 17.1 * (1.0 - 2.2557e-5 * h_m) ** 5.2559

    el_rad = math.radians(elev_deg)
    # Saastamoinen zenith delay formula
    zenith_delay = 0.002277 * (P + (1255.0 / T + 0.05) * e)
    # Simple 1/sin mapping function (adequate for elev > 10°)
    slant_delay = zenith_delay / math.sin(el_rad)
    return slant_delay


# ---------------------------------------------------------------------------
# [NEW] Klobuchar ionospheric correction
# ---------------------------------------------------------------------------

@dataclass
class KlobucharParams:
    """Eight Klobuchar alpha/beta coefficients broadcast in GPS nav files."""
    alpha: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    beta:  Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def klobuchar_iono_delay_m(
    params: KlobucharParams,
    rx_lat_deg: float,
    rx_lon_deg: float,
    elev_deg: float,
    azimuth_deg: float,
    gps_tow_s: float,
) -> float:
    """
    Klobuchar single-frequency ionospheric delay model (IS-GPS-200).

    Returns the L1 ionospheric delay in metres (positive = delay).
    Typical correction removes 50-75% of the actual delay.

    Parameters
    ----------
    params      : KlobucharParams with alpha/beta coefficients
    rx_lat_deg  : receiver geodetic latitude, degrees
    rx_lon_deg  : receiver longitude, degrees
    elev_deg    : satellite elevation angle, degrees
    azimuth_deg : satellite azimuth angle, degrees (measured from North, clockwise)
    gps_tow_s   : GPS time of week, seconds
    """
    if elev_deg < 0.0:
        elev_deg = 0.0

    el_sc = elev_deg / 180.0          # elevation in semi-circles
    az_rad = math.radians(azimuth_deg)
    lat_u = rx_lat_deg / 180.0        # user lat in semi-circles

    # Earth-centred angle (semi-circles)
    psi = 0.0137 / (el_sc + 0.11) - 0.022

    # Subionospheric point latitude (semi-circles)
    lat_i = lat_u + psi * math.cos(az_rad)
    lat_i = max(-0.416, min(0.416, lat_i))

    # Subionospheric point longitude (semi-circles)
    lon_i = rx_lon_deg / 180.0 + psi * math.sin(az_rad) / math.cos(math.radians(lat_i * 180.0))

    # Geomagnetic latitude of subionospheric point (semi-circles)
    lat_m = lat_i + 0.064 * math.cos(math.radians((lon_i - 1.617) * 180.0))

    # Local time at subionospheric point
    t = 4.32e4 * lon_i + gps_tow_s
    t = t % 86400.0

    # Period and amplitude of the cosine term
    PER = sum(params.beta[n] * (lat_m ** n) for n in range(4))
    PER = max(72000.0, PER)

    AMP = sum(params.alpha[n] * (lat_m ** n) for n in range(4))
    AMP = max(0.0, AMP)

    x = 2.0 * math.pi * (t - 50400.0) / PER

    # Vertical delay (seconds)
    if abs(x) < 1.57:
        F_vert = 5e-9 + AMP * (1.0 - x*x/2.0 + x**4/24.0)
    else:
        F_vert = 5e-9

    # Obliquity factor
    F_obliq = 1.0 + 16.0 * (0.53 - el_sc) ** 3

    delay_s = F_obliq * F_vert
    return delay_s * C   # convert to metres


def azimuth_angle_deg(rx_ecef: np.ndarray, sv_ecef: np.ndarray) -> float:
    """Azimuth of satellite from receiver, degrees clockwise from North."""
    lat, lon, _ = ecef_to_geodetic(*rx_ecef.tolist())
    enu_rot = ecef_to_enu_matrix(lat, lon)
    los = sv_ecef - rx_ecef
    los_unit = los / np.linalg.norm(los)
    enu = enu_rot @ los_unit
    az = math.degrees(math.atan2(enu[0], enu[1]))   # atan2(E, N)
    return az % 360.0


def gps_tow_from_datetime(t: dt.datetime) -> float:
    """Return GPS time-of-week in seconds for a datetime in GPS time."""
    GPS_EPOCH = dt.datetime(1980, 1, 6, 0, 0, 0)
    total_seconds = (t - GPS_EPOCH).total_seconds()
    return total_seconds % (7 * 86400.0)


# ---------------------------------------------------------------------------
# [NEW] RINEX NAV parser (Klobuchar coefficients only)
# ---------------------------------------------------------------------------

def parse_klobuchar_from_nav(path: str) -> Optional[KlobucharParams]:
    """
    Extract Klobuchar alpha/beta coefficients from a RINEX 3/4 nav file.

    Looks for IONOSPHERIC CORR header records of type GPSA and GPSB,
    or the older RINEX 2 ION ALPHA / ION BETA lines.
    Returns None if coefficients are not found.
    """
    alpha: Optional[List[float]] = None
    beta:  Optional[List[float]] = None

    def _split_coeffs(s: str) -> List[float]:
        """Parse up to 4 space-separated D-notation floats."""
        vals = []
        for tok in s.split():
            v = parse_float(tok)
            if v is not None:
                vals.append(v)
            if len(vals) == 4:
                break
        return vals

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                label = line[60:].strip() if len(line) >= 61 else ""

                # RINEX 3/4 style
                if label == "IONOSPHERIC CORR":
                    corr_type = line[:4].strip()
                    coeffs = _split_coeffs(line[5:60])
                    if corr_type == "GPSA" and len(coeffs) == 4:
                        alpha = coeffs
                    elif corr_type == "GPSB" and len(coeffs) == 4:
                        beta = coeffs

                # RINEX 2 style
                elif label == "ION ALPHA":
                    alpha = _split_coeffs(line[:60])
                elif label == "ION BETA":
                    beta = _split_coeffs(line[:60])

                elif label == "END OF HEADER":
                    break

    except OSError as e:
        print(f"[WARN] Could not open NAV file: {e}", file=sys.stderr)
        return None

    if alpha and beta and len(alpha) == 4 and len(beta) == 4:
        return KlobucharParams(
            alpha=tuple(alpha),   # type: ignore[arg-type]
            beta=tuple(beta),     # type: ignore[arg-type]
        )

    print("[WARN] Klobuchar coefficients not found in NAV file; ionospheric correction disabled.",
          file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# [NEW] IGS precise clock file parser
# ---------------------------------------------------------------------------

class PreciseClockInterpolator:
    """
    Parses an IGS RINEX clock file (.clk) and provides linear interpolation
    of satellite clock corrections.

    IGS clock files are typically at 30 s or 5 s intervals.  Linear
    interpolation between adjacent epochs introduces < 0.1 ns of error
    for a 30 s file (< 3 cm), which is far better than Lagrange-interpolated
    SP3 clocks at 15-minute spacing (~1–3 m).

    Only AS (satellite clock) records are used; AR (receiver) records are ignored.
    Satellite IDs use the standard 3-char RINEX convention (e.g. "G01").
    Clock values with |bias| >= 0.999 s are treated as bad/missing.
    """

    CLK_SENTINEL = 0.999  # seconds; values at or above this are invalid

    def __init__(self, path: str):
        # sat_id -> sorted list of (epoch_datetime, bias_s)
        self._data: Dict[str, List[Tuple[dt.datetime, float]]] = {}
        self._parse(path)

    def _parse(self, path: str) -> None:
        in_header = True
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if in_header:
                    if len(line) >= 60 and line[60:].strip() == "END OF HEADER":
                        in_header = False
                    continue

                # Data record format (RINEX clock):
                # AS G01 2024 001 00 00 00.000000  1  -1.234567890123D-04  1.234D-12
                if not line.startswith("AS"):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                sat_id = parts[1]
                try:
                    year   = int(parts[2])
                    month  = int(parts[3])
                    day    = int(parts[4])
                    hour   = int(parts[5])
                    minute = int(parts[6])
                    sec    = float(parts[7])
                    # number of values is parts[8]; bias is parts[9]
                    if len(parts) < 10:
                        continue
                    bias_s = parse_float(parts[9])
                except (ValueError, IndexError):
                    continue

                if bias_s is None or abs(bias_s) >= self.CLK_SENTINEL:
                    continue

                epoch = datetime_from_rinex_fields(year, month, day, hour, minute, sec)
                self._data.setdefault(sat_id, []).append((epoch, bias_s))

        # Sort each satellite's list by epoch
        for sat_id in self._data:
            self._data[sat_id].sort(key=lambda x: x[0])

        n_sats = len(self._data)
        if n_sats == 0:
            print("[WARN] Precise clock file contained no valid AS records.", file=sys.stderr)
        else:
            print(f"[INFO] Precise clock file loaded: {n_sats} satellites.", file=sys.stderr)

    def get_clock_s(self, sat_id: str, t: dt.datetime) -> Optional[float]:
        """
        Return linearly interpolated satellite clock bias in seconds at time t.
        Returns None if the satellite is not in the file or t is out of range.
        """
        records = self._data.get(sat_id)
        if not records:
            return None

        times = [r[0] for r in records]

        # Binary search for bracket
        lo, hi = 0, len(records) - 1
        if t <= times[lo]:
            return records[lo][1]
        if t >= times[hi]:
            return records[hi][1]

        # Find index where times[idx] <= t < times[idx+1]
        idx = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if times[mid] <= t:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        t0, c0 = records[idx]
        t1, c1 = records[idx + 1]
        dt_span = (t1 - t0).total_seconds()
        if dt_span <= 0.0:
            return c0
        frac = (t - t0).total_seconds() / dt_span
        return c0 + frac * (c1 - c0)


# ---------------------------------------------------------------------------
# RINEX observation parser (unchanged from v1)
# ---------------------------------------------------------------------------

class RinexObsParser:
    PREFERRED_CODE_OBS = ["C1C", "C1W", "C1P", "C1S", "C1L", "C1X"]

    def __init__(self, path: str):
        self.path = path
        self.obs_types_by_sys: Dict[str, List[str]] = {}

    def parse(self) -> List[EpochObs]:
        epochs: List[EpochObs] = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            self._parse_header(f)
            current_epoch: Optional[EpochObs] = None

            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    current_epoch = self._parse_epoch_header(line)
                    if current_epoch is not None:
                        epochs.append(current_epoch)
                    continue

                if current_epoch is None:
                    continue

                sat_obs = self._parse_sat_line(line)
                if sat_obs is not None:
                    current_epoch.sats.append(sat_obs)

        return epochs

    def _parse_header(self, f) -> None:
        pending_sys = None
        pending_types: List[str] = []

        for raw_line in f:
            line = raw_line.rstrip("\n")
            label = line[60:].strip() if len(line) >= 61 else ""

            if label == "SYS / # / OBS TYPES":
                sys_char = line[0]
                count = int(line[3:6].strip())
                types_here = line[7:60].split()

                if pending_sys is None or pending_sys != sys_char:
                    pending_sys = sys_char
                    pending_types = []

                pending_types.extend(types_here)

                if len(pending_types) >= count:
                    self.obs_types_by_sys[pending_sys] = pending_types[:count]
                    pending_sys = None
                    pending_types = []

            elif label == "END OF HEADER":
                return

        raise ValueError("RINEX header ended unexpectedly without END OF HEADER")

    def _parse_epoch_header(self, line: str) -> Optional[EpochObs]:
        parts = line[1:].split()
        if len(parts) < 8:
            return None

        year   = int(parts[0])
        month  = int(parts[1])
        day    = int(parts[2])
        hour   = int(parts[3])
        minute = int(parts[4])
        second = float(parts[5])
        time_gps = datetime_from_rinex_fields(year, month, day, hour, minute, second)
        return EpochObs(time_gps=time_gps)

    def _parse_sat_line(self, line: str) -> Optional[SatObs]:
        if len(line) < 3:
            return None

        sat_id = line[:3].strip()
        if not sat_id or sat_id[0] != "G":
            return None

        obs_types = self.obs_types_by_sys.get("G")
        if not obs_types:
            return None

        values: Dict[str, Optional[float]] = {}
        fields = line[3:]

        for idx, obs_name in enumerate(obs_types):
            start = idx * 16
            if start >= len(fields):
                values[obs_name] = None
                continue
            chunk = fields[start:start + 16]
            val = parse_float(chunk[:14])
            values[obs_name] = val

        pseudorange = None
        pr_obs_name = None
        for obs_name in self.PREFERRED_CODE_OBS:
            val = values.get(obs_name)
            if val is not None and val > 1e6:
                pseudorange = val
                pr_obs_name = obs_name
                break

        if pseudorange is None:
            return None

        snr = None
        doppler = None
        if pr_obs_name and pr_obs_name.startswith("C1"):
            suffix = pr_obs_name[2:]
            snr = values.get("S1" + suffix)
            doppler = values.get("D1" + suffix)

        return SatObs(sat_id=sat_id, pseudorange_m=pseudorange, doppler_hz=doppler, snr=snr)


# ---------------------------------------------------------------------------
# SP3 parser & Lagrange interpolator (unchanged from v1, with sentinel fix)
# ---------------------------------------------------------------------------

class Sp3Parser:
    def __init__(self, path: str):
        self.path = path

    def parse(self) -> List[Sp3Record]:
        records: List[Sp3Record] = []
        current: Optional[Sp3Record] = None

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue

                if line.startswith("*"):
                    parts = line[1:].split()
                    year   = int(parts[0])
                    month  = int(parts[1])
                    day    = int(parts[2])
                    hour   = int(parts[3])
                    minute = int(parts[4])
                    second = float(parts[5])
                    t = datetime_from_rinex_fields(year, month, day, hour, minute, second)
                    current = Sp3Record(time=t)
                    records.append(current)

                elif line.startswith("P") and current is not None:
                    sat_id = line[1:4].strip()
                    x_km = parse_float(line[4:18])
                    y_km = parse_float(line[18:32])
                    z_km = parse_float(line[32:46])
                    clk_us = parse_float(line[46:60])

                    if x_km is None or y_km is None or z_km is None:
                        continue

                    current.sat_positions_m[sat_id] = np.array([
                        x_km * 1000.0,
                        y_km * 1000.0,
                        z_km * 1000.0,
                    ], dtype=float)

                    # [FIX] Filter SP3 clock sentinel values (999999.999999 µs)
                    if clk_us is not None and abs(clk_us) < 999990.0:
                        current.sat_clocks_s[sat_id] = clk_us * 1e-6

        if not records:
            raise ValueError(f"No SP3 epochs found in {self.path}")
        return records


class Sp3Interpolator:
    """
    Interpolates satellite position and optional SP3 clock values.

    Uses multi-point Lagrange (barycentric) interpolation.  SP3 clocks are
    used only as a fallback when no precise clock file is available.

    Recommended interpolation_points values: 9, 11 (default), or 13.
    """

    def __init__(self, records: List[Sp3Record], interpolation_points: int = 11):
        if interpolation_points < 2:
            raise ValueError("interpolation_points must be at least 2")
        self.records = records
        self.times = [r.time for r in records]
        self.interpolation_points = interpolation_points

    def interpolate(self, sat_id: str, t: dt.datetime) -> Tuple[np.ndarray, Optional[float]]:
        records = self._select_records_for_sat(sat_id, t, self.interpolation_points)

        if len(records) == 1:
            return self._from_single_record(records[0], sat_id)

        t_ref = records[0].time
        x_query = (t - t_ref).total_seconds()
        xs = np.array([(r.time - t_ref).total_seconds() for r in records], dtype=float)

        pos_values = np.array([r.sat_positions_m[sat_id] for r in records], dtype=float)
        pos = self._lagrange_barycentric(xs, pos_values, x_query)

        clk = None
        if all(sat_id in r.sat_clocks_s for r in records):
            clk_values = np.array([r.sat_clocks_s[sat_id] for r in records], dtype=float)
            clk = float(self._lagrange_barycentric(xs, clk_values, x_query))

        return pos, clk

    def _select_records_for_sat(
        self,
        sat_id: str,
        t: dt.datetime,
        desired_points: int,
    ) -> List[Sp3Record]:
        n = len(self.records)
        if n == 0:
            raise KeyError("No SP3 records loaded")

        desired_points = min(desired_points, n)

        indices = list(range(n))
        indices.sort(key=lambda i: abs((self.times[i] - t).total_seconds()))

        selected = []
        for i in indices:
            if sat_id in self.records[i].sat_positions_m:
                selected.append(i)
                if len(selected) >= desired_points:
                    break

        if not selected:
            raise KeyError(f"{sat_id} not present in SP3 near {t.isoformat()}")

        selected.sort()
        return [self.records[i] for i in selected]

    @staticmethod
    def _lagrange_barycentric(xs: np.ndarray, ys: np.ndarray, x: float):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        n = xs.size

        diff = x - xs
        hit = np.where(np.abs(diff) < 1e-9)[0]
        if hit.size:
            return ys[hit[0]]

        weights = np.ones(n, dtype=float)
        for j in range(n):
            denom = 1.0
            for m in range(n):
                if m != j:
                    denom *= (xs[j] - xs[m])
            weights[j] = 1.0 / denom

        terms = weights / diff

        if ys.ndim == 1:
            return np.sum(terms * ys) / np.sum(terms)

        numerator = np.sum(terms[:, None] * ys, axis=0)
        denominator = np.sum(terms)
        return numerator / denominator

    @staticmethod
    def _from_single_record(record: Sp3Record, sat_id: str) -> Tuple[np.ndarray, Optional[float]]:
        if sat_id not in record.sat_positions_m:
            raise KeyError(f"{sat_id} not present in SP3 record at {record.time.isoformat()}")
        return record.sat_positions_m[sat_id], record.sat_clocks_s.get(sat_id)


# ---------------------------------------------------------------------------
# Epoch position solver
# ---------------------------------------------------------------------------

def solve_epoch_position(
    epoch: EpochObs,
    sp3: Sp3Interpolator,
    initial_rx_ecef: Optional[np.ndarray] = None,
    initial_clock_bias_m: float = 0.0,
    min_snr: Optional[float] = None,
    elevation_mask_deg: Optional[float] = None,
    max_iterations: int = 10,
    convergence_tol_m: float = 1e-3,
    residual_reject_threshold_m: Optional[float] = 80.0,
    precise_clocks: Optional[PreciseClockInterpolator] = None,   # [NEW]
    klobuchar: Optional[KlobucharParams] = None,                 # [NEW]
) -> Optional[Tuple[np.ndarray, float, float, List[str]]]:
    """
    Solve receiver ECEF position and clock bias for one epoch.

    Returns (rx_ecef, clock_bias_m, rms_residual_m, used_sat_ids) or None.

    New parameters
    --------------
    precise_clocks : if provided, satellite clocks are taken from the IGS CLK
                     file (with SP3 as fallback).  This is the single biggest
                     accuracy improvement available.
    klobuchar      : if provided, the Klobuchar ionospheric model is applied
                     per satellite.
    """
    sats = [s for s in epoch.sats if s.pseudorange_m > 1e6]
    if min_snr is not None:
        sats = [s for s in sats if s.snr is None or s.snr >= min_snr]

    if len(sats) < 4:
        return None

    if initial_rx_ecef is None:
        rx = np.array([WGS84_A, 0.0, 0.0], dtype=float)
    else:
        rx = initial_rx_ecef.astype(float).copy()

    cb = float(initial_clock_bias_m)
    used_sat_ids: List[str] = []

    for _ in range(max_iterations):
        H_rows: List[np.ndarray] = []
        residuals: List[float] = []
        candidate_sat_ids: List[str] = []

        for sat in sats:
            transit_time = sat.pseudorange_m / C
            tx_time = epoch.time_gps - dt.timedelta(seconds=transit_time)

            try:
                sv_pos, sv_clk_s_sp3 = sp3.interpolate(sat.sat_id, tx_time)
            except KeyError:
                continue

            sv_pos_corr = apply_earth_rotation_correction(sv_pos, transit_time)

            if elevation_mask_deg is not None and np.linalg.norm(rx) > 1e3:
                elev = elevation_angle_deg(rx, sv_pos_corr)
                if elev < elevation_mask_deg:
                    continue

            rho_geom = np.linalg.norm(sv_pos_corr - rx)

            # ----------------------------------------------------------------
            # [NEW] Satellite clock: prefer precise CLK file, fall back to SP3
            # ----------------------------------------------------------------
            if precise_clocks is not None:
                sv_clk_s = precise_clocks.get_clock_s(sat.sat_id, tx_time)
                if sv_clk_s is None:
                    sv_clk_s = sv_clk_s_sp3   # fallback
            else:
                sv_clk_s = sv_clk_s_sp3

            sv_clk_m = C * sv_clk_s if sv_clk_s is not None else 0.0

            # ----------------------------------------------------------------
            # [NEW] Tropospheric correction (Saastamoinen, standard atmosphere)
            # ----------------------------------------------------------------
            elev_for_corr = elevation_angle_deg(rx, sv_pos_corr) if np.linalg.norm(rx) > 1e3 else 90.0
            _, _, h_rx = ecef_to_geodetic(*rx.tolist()) if np.linalg.norm(rx) > 1e3 else (0.0, 0.0, 0.0)
            tropo_m = saastamoinen_tropo_delay_m(max(0.0, h_rx), elev_for_corr)

            # ----------------------------------------------------------------
            # [NEW] Ionospheric correction (Klobuchar, optional)
            # ----------------------------------------------------------------
            iono_m = 0.0
            if klobuchar is not None and np.linalg.norm(rx) > 1e3:
                lat_rx, lon_rx, _ = ecef_to_geodetic(*rx.tolist())
                az = azimuth_angle_deg(rx, sv_pos_corr)
                tow = gps_tow_from_datetime(epoch.time_gps)
                iono_m = klobuchar_iono_delay_m(klobuchar, lat_rx, lon_rx, elev_for_corr, az, tow)

            # Predicted pseudorange (positive corrections = range increase)
            pred = rho_geom + cb - sv_clk_m + tropo_m + iono_m
            v = sat.pseudorange_m - pred

            los = (rx - sv_pos_corr) / rho_geom
            H = np.array([los[0], los[1], los[2], 1.0], dtype=float)

            H_rows.append(H)
            residuals.append(v)
            candidate_sat_ids.append(sat.sat_id)

        if len(H_rows) < 4:
            return None

        H_mat = np.vstack(H_rows)
        y = np.array(residuals, dtype=float)

        try:
            dx, *_ = np.linalg.lstsq(H_mat, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        rx += dx[:3]
        cb += dx[3]
        used_sat_ids = candidate_sat_ids

        if np.linalg.norm(dx[:3]) < convergence_tol_m and abs(dx[3]) < convergence_tol_m:
            break

    # -----------------------------------------------------------------------
    # Final residual pass + optional outlier rejection (unchanged from v1)
    # -----------------------------------------------------------------------
    final_residuals = []
    final_rows = []
    for sat in sats:
        transit_time = sat.pseudorange_m / C
        tx_time = epoch.time_gps - dt.timedelta(seconds=transit_time)
        try:
            sv_pos, sv_clk_s_sp3 = sp3.interpolate(sat.sat_id, tx_time)
        except KeyError:
            continue
        sv_pos_corr = apply_earth_rotation_correction(sv_pos, transit_time)

        if elevation_mask_deg is not None:
            elev = elevation_angle_deg(rx, sv_pos_corr)
            if elev < elevation_mask_deg:
                continue

        rho_geom = np.linalg.norm(sv_pos_corr - rx)

        if precise_clocks is not None:
            sv_clk_s = precise_clocks.get_clock_s(sat.sat_id, tx_time)
            if sv_clk_s is None:
                sv_clk_s = sv_clk_s_sp3
        else:
            sv_clk_s = sv_clk_s_sp3

        sv_clk_m = C * sv_clk_s if sv_clk_s is not None else 0.0

        elev_fc = elevation_angle_deg(rx, sv_pos_corr)
        _, _, h_rx = ecef_to_geodetic(*rx.tolist())
        tropo_m = saastamoinen_tropo_delay_m(max(0.0, h_rx), elev_fc)

        iono_m = 0.0
        if klobuchar is not None:
            lat_rx, lon_rx, _ = ecef_to_geodetic(*rx.tolist())
            az = azimuth_angle_deg(rx, sv_pos_corr)
            tow = gps_tow_from_datetime(epoch.time_gps)
            iono_m = klobuchar_iono_delay_m(klobuchar, lat_rx, lon_rx, elev_fc, az, tow)

        pred = rho_geom + cb - sv_clk_m + tropo_m + iono_m
        res = sat.pseudorange_m - pred

        final_residuals.append(res)
        final_rows.append((sat, sv_pos_corr))

    if len(final_residuals) < 4:
        return None

    if residual_reject_threshold_m is not None:
        keep_idx = [i for i, r in enumerate(final_residuals) if abs(r) <= residual_reject_threshold_m]
        if len(keep_idx) >= 4 and len(keep_idx) < len(final_residuals):
            filtered_epoch = EpochObs(time_gps=epoch.time_gps, sats=[final_rows[i][0] for i in keep_idx])
            return solve_epoch_position(
                filtered_epoch,
                sp3,
                initial_rx_ecef=rx,
                initial_clock_bias_m=cb,
                min_snr=min_snr,
                elevation_mask_deg=elevation_mask_deg,
                max_iterations=max_iterations,
                convergence_tol_m=convergence_tol_m,
                residual_reject_threshold_m=None,
                precise_clocks=precise_clocks,
                klobuchar=klobuchar,
            )

    rms = float(np.sqrt(np.mean(np.square(final_residuals))))
    final_sat_ids = [row[0].sat_id for row in final_rows]
    return rx, cb, rms, final_sat_ids


# ---------------------------------------------------------------------------
# Velocity (unchanged from v1)
# ---------------------------------------------------------------------------

def add_velocity_columns(solutions: List[SolutionEpoch]) -> None:
    if len(solutions) < 2:
        return

    times = np.array([s.time_gps.timestamp() for s in solutions], dtype=float)
    xyz = np.array([[s.x_m, s.y_m, s.z_m] for s in solutions], dtype=float)
    vel = np.zeros_like(xyz)

    for i in range(len(solutions)):
        if i == 0:
            dt_s = times[i + 1] - times[i]
            if dt_s > 0:
                vel[i] = (xyz[i + 1] - xyz[i]) / dt_s
        elif i == len(solutions) - 1:
            dt_s = times[i] - times[i - 1]
            if dt_s > 0:
                vel[i] = (xyz[i] - xyz[i - 1]) / dt_s
        else:
            dt_s = times[i + 1] - times[i - 1]
            if dt_s > 0:
                vel[i] = (xyz[i + 1] - xyz[i - 1]) / dt_s

    for s, v in zip(solutions, vel):
        s.vx_mps = float(v[0])
        s.vy_mps = float(v[1])
        s.vz_mps = float(v[2])
        s.speed_mps = float(np.linalg.norm(v))


# ---------------------------------------------------------------------------
# Output writers (unchanged from v1)
# ---------------------------------------------------------------------------

def write_csv(path: str, solutions: Sequence[SolutionEpoch]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "utc_time", "gps_time",
            "x_ecef_m", "y_ecef_m", "z_ecef_m",
            "lat_deg", "lon_deg", "height_m",
            "vx_mps", "vy_mps", "vz_mps", "speed_mps",
            "clock_bias_m", "num_sats", "rms_residual_m",
        ])
        for s in solutions:
            writer.writerow([
                s.time_utc.isoformat(),
                s.time_gps.isoformat(),
                f"{s.x_m:.4f}", f"{s.y_m:.4f}", f"{s.z_m:.4f}",
                f"{s.lat_deg:.9f}", f"{s.lon_deg:.9f}", f"{s.h_m:.4f}",
                f"{s.vx_mps:.4f}", f"{s.vy_mps:.4f}", f"{s.vz_mps:.4f}", f"{s.speed_mps:.4f}",
                f"{s.clock_bias_m:.4f}", s.num_sats, f"{s.rms_residual_m:.4f}",
            ])


def write_kml(path: str, solutions: Sequence[SolutionEpoch], name: str = "GNSS Offline Path") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    coords = "\n".join(
        f"          {s.lon_deg:.9f},{s.lat_deg:.9f},{s.h_m:.3f}"
        for s in solutions
    )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>{name}</name>\n'
        '    <Placemark>\n'
        f'      <name>{name}</name>\n'
        '      <LineString>\n'
        '        <tessellate>1</tessellate>\n'
        '        <altitudeMode>absolute</altitudeMode>\n'
        '        <coordinates>\n'
        f'{coords}\n'
        '        </coordinates>\n'
        '      </LineString>\n'
        '    </Placemark>\n'
        '  </Document>\n'
        '</kml>\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(kml)


# ---------------------------------------------------------------------------
# 1 Hz decimation (unchanged from v1)
# ---------------------------------------------------------------------------

def decimate_epochs_to_1hz(epochs: Sequence[EpochObs]) -> List[EpochObs]:
    out: List[EpochObs] = []
    seen = set()
    for ep in epochs:
        key = ep.time_gps.replace(microsecond=0)
        if key in seen:
            continue
        seen.add(key)
        out.append(ep)
    return out



# ---------------------------------------------------------------------------
# [NEW] Automatic SP3 / CLK download using gnss-lib-py
# ---------------------------------------------------------------------------

def gps_datetime_to_gps_millis(gps_dt: dt.datetime) -> float:
    """
    Convert a naive datetime representing GPS time to GPS milliseconds.

    RINEX observation epochs in this script are treated as GPS time.
    gnss-lib-py's ephemeris downloader expects gps_millis.
    """
    gps_epoch = dt.datetime(1980, 1, 6, 0, 0, 0)
    return (gps_dt - gps_epoch).total_seconds() * 1000.0


def _normalize_ephemeris_paths(paths_obj) -> List[str]:
    """
    Normalize gnss-lib-py downloader outputs into a List[str].
    """
    if paths_obj is None:
        return []

    if isinstance(paths_obj, (str, os.PathLike)):
        return [str(paths_obj)]

    if isinstance(paths_obj, (list, tuple)):
        out = []
        for item in paths_obj:
            if isinstance(item, (list, tuple)):
                out.extend(_normalize_ephemeris_paths(item))
            else:
                out.append(str(item))
        return out

    return [str(paths_obj)]


def auto_download_ephemeris_with_gnss_lib_py(
    epochs: Sequence[EpochObs],
    file_type: str,
    verbose: bool = True,
) -> List[str]:
    """
    Use gnss-lib-py only as an ephemeris downloader.

    This script still parses RINEX 4.x by itself. We only give gnss-lib-py
    the recording time range so it can select/download the correct SP3/CLK file.

    file_type:
        "sp3" -> precise orbit file
        "clk" -> precise clock file
    """
    if not epochs:
        raise ValueError("Cannot download ephemeris because no RINEX epochs were parsed.")

    try:
        import gnss_lib_py as glp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Automatic ephemeris download requires gnss-lib-py.\n"
            "Install it with:\n"
            "    pip install gnss-lib-py\n"
            "Note: gnss-lib-py may require Python < 3.13."
        ) from exc

    # A small buffer protects recordings close to day boundaries.
    first = epochs[0].time_gps - dt.timedelta(minutes=30)
    last = epochs[-1].time_gps + dt.timedelta(minutes=30)

    gps_millis = np.array([
        gps_datetime_to_gps_millis(first),
        gps_datetime_to_gps_millis(last),
    ], dtype=float)

    if verbose:
        print(
            f"[INFO] Auto-downloading {file_type.upper()} using gnss-lib-py "
            f"for GPS time range {first.isoformat()} -> {last.isoformat()}",
            file=sys.stderr,
        )

    try:
        paths_obj = glp.load_ephemeris(
            file_type=file_type,
            gps_millis=gps_millis,
            verbose=verbose,
        )
    except AttributeError:
        try:
            from gnss_lib_py.utils.ephemeris_downloader import load_ephemeris  # type: ignore
            paths_obj = load_ephemeris(
                file_type=file_type,
                gps_millis=gps_millis,
                verbose=verbose,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not call gnss-lib-py's load_ephemeris function. "
                "Your installed gnss-lib-py version may expose a different API."
            ) from exc

    paths = _normalize_ephemeris_paths(paths_obj)

    if not paths:
        raise RuntimeError(f"gnss-lib-py did not return any {file_type.upper()} file paths.")

    if verbose:
        print(f"[INFO] gnss-lib-py returned {file_type.upper()} file(s):", file=sys.stderr)
        for p in paths:
            print(f"       {p}", file=sys.stderr)

    return paths


def choose_first_existing_path(paths: Sequence[str], label: str) -> str:
    """
    Choose the first returned path that exists locally.

    If a short recording spans one day, this is enough. If your recording spans
    midnight and gnss-lib-py returns multiple SP3 files, the current parser uses
    the first file only. That is fine for normal short recordings, but a long
    cross-midnight recording should be handled by merging SP3 records.
    """
    for p in paths:
        if os.path.exists(p):
            return p

    if paths:
        print(
            f"[WARN] None of the returned {label} paths were found with os.path.exists; "
            f"using first returned path anyway: {paths[0]}",
            file=sys.stderr,
        )
        return paths[0]

    raise RuntimeError(f"No {label} paths were available.")


# ---------------------------------------------------------------------------
# Top-level path solver
# ---------------------------------------------------------------------------

def solve_path(
    obs_path: str,
    sp3_path: Optional[str],
    gps_utc_leap_seconds: int,
    min_snr: Optional[float] = None,
    elevation_mask_deg: Optional[float] = None,
    sp3_interp_points: int = 11,
    decimate_to_1hz: bool = True,
    clk_path: Optional[str] = None,      # [NEW] IGS precise clock file
    nav_path: Optional[str] = None,      # [NEW] RINEX NAV file for Klobuchar
    auto_clk: bool = False,              # [NEW] auto-download CLK with gnss-lib-py
) -> List[SolutionEpoch]:
    obs_parser = RinexObsParser(obs_path)
    epochs = obs_parser.parse()
    if decimate_to_1hz:
        epochs = decimate_epochs_to_1hz(epochs)

    if not epochs:
        raise ValueError("No RINEX observation epochs were parsed.")

    # [NEW] If --sp3 was not provided, automatically download the correct SP3.
    if sp3_path is None:
        sp3_paths = auto_download_ephemeris_with_gnss_lib_py(
            epochs=epochs,
            file_type="sp3",
            verbose=True,
        )
        sp3_path = choose_first_existing_path(sp3_paths, "SP3")

    print(f"[INFO] Using SP3 file: {sp3_path}", file=sys.stderr)
    sp3_records = Sp3Parser(sp3_path).parse()
    sp3 = Sp3Interpolator(sp3_records, interpolation_points=sp3_interp_points)

    # [NEW] Optionally auto-download precise CLK if --clk was not provided.
    if auto_clk and clk_path is None:
        try:
            clk_paths = auto_download_ephemeris_with_gnss_lib_py(
                epochs=epochs,
                file_type="clk",
                verbose=True,
            )
            clk_path = choose_first_existing_path(clk_paths, "CLK")
        except Exception as exc:
            print(f"[WARN] Automatic CLK download failed: {exc}", file=sys.stderr)
            print("[WARN] Continuing without precise CLK file.", file=sys.stderr)

    # [NEW] Load optional precise clock file
    precise_clocks: Optional[PreciseClockInterpolator] = None
    if clk_path:
        print(f"[INFO] Loading precise clock file: {clk_path}", file=sys.stderr)
        precise_clocks = PreciseClockInterpolator(clk_path)

    # [NEW] Load optional Klobuchar parameters from NAV file
    klobuchar: Optional[KlobucharParams] = None
    if nav_path:
        print(f"[INFO] Loading Klobuchar coefficients from: {nav_path}", file=sys.stderr)
        klobuchar = parse_klobuchar_from_nav(nav_path)

    solutions: List[SolutionEpoch] = []
    prev_rx = None
    prev_cb = 0.0

    for idx, epoch in enumerate(epochs, start=1):
        sol = solve_epoch_position(
            epoch=epoch,
            sp3=sp3,
            initial_rx_ecef=prev_rx,
            initial_clock_bias_m=prev_cb,
            min_snr=min_snr,
            elevation_mask_deg=elevation_mask_deg,
            precise_clocks=precise_clocks,
            klobuchar=klobuchar,
        )
        if sol is None:
            print(f"[WARN] epoch {idx}/{len(epochs)} {epoch.time_gps.isoformat()} -> no solution",
                  file=sys.stderr)
            continue

        rx, cb, rms, used_sat_ids = sol
        lat, lon, h = ecef_to_geodetic(*rx.tolist())
        utc_time = gps_datetime_to_utc(epoch.time_gps, gps_utc_leap_seconds)

        solutions.append(SolutionEpoch(
            time_gps=epoch.time_gps,
            time_utc=utc_time,
            x_m=float(rx[0]),
            y_m=float(rx[1]),
            z_m=float(rx[2]),
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            clock_bias_m=cb,
            num_sats=len(used_sat_ids),
            rms_residual_m=rms,
        ))

        prev_rx = rx
        prev_cb = cb

    add_velocity_columns(solutions)
    return solutions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline GNSS path solver from RINEX OBS + SP3 precise orbits"
    )
    p.add_argument("--obs", required=True,
                   help="Path to RINEX observation file (.o/.obs)")
    p.add_argument("--sp3", default=None,
                   help="Optional path to local SP3 precise orbit file. "
                        "If omitted, the script uses gnss-lib-py to auto-download the needed SP3.")
    p.add_argument("--clk", default=None,
                   help="[NEW] Optional path to IGS precise clock file (.clk). "
                        "If omitted, use --auto-clk to try automatic CLK download.")
    p.add_argument("--auto-clk", action="store_true",
                   help="[NEW] Try to auto-download a precise CLK file with gnss-lib-py.")
    p.add_argument("--nav", default=None,
                   help="[NEW] Path to RINEX NAV file for Klobuchar ionospheric correction. "
                        "Reduces single-frequency iono error by ~50–75%%.")
    p.add_argument("--out-prefix", required=True,
                   help="Output prefix, e.g. results/track")
    p.add_argument("--gps-utc-leap-seconds", type=int, default=18,
                   help="GPS-UTC leap seconds (default: 18, correct for 2017-present)")
    p.add_argument("--min-snr", type=float, default=None,
                   help="Optional minimum SNR threshold (dB-Hz)")
    p.add_argument("--elevation-mask-deg", type=float, default=None,
                   help="Optional elevation mask in degrees (recommended: 10–15)")
    p.add_argument("--sp3-interp-points", type=int, default=11,
                   help="Number of SP3 epochs for Lagrange interpolation (default: 11)")
    p.add_argument("--no-decimate", action="store_true",
                   help="Do not decimate to 1 Hz")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    solutions = solve_path(
        obs_path=args.obs,
        sp3_path=args.sp3,
        gps_utc_leap_seconds=args.gps_utc_leap_seconds,
        min_snr=args.min_snr,
        elevation_mask_deg=args.elevation_mask_deg,
        sp3_interp_points=args.sp3_interp_points,
        decimate_to_1hz=not args.no_decimate,
        clk_path=args.clk,
        nav_path=args.nav,
        auto_clk=args.auto_clk,
    )

    if not solutions:
        print("No valid solution epochs were produced.", file=sys.stderr)
        return 1

    csv_path = args.out_prefix + ".csv"
    kml_path = args.out_prefix + ".kml"
    write_csv(csv_path, solutions)
    write_kml(kml_path, solutions)

    print(f"Wrote {len(solutions)} solution epochs")
    print(f"CSV: {csv_path}")
    print(f"KML: {kml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
