#!/usr/bin/env python3
"""
gnss_offline_path.py

A practical first-pass GNSS offline path solver for:
- RINEX 4.x observation files (GPS-only, pseudorange C1C/C1W/C1P/C1S/C1L/C1X)
- SP3 precise orbit files (Lagrange polynomial interpolation)
- 1 Hz trajectory output to CSV + KML

Designed as a course-project baseline:
- parses observation epochs
- filters satellites and observations
- interpolates satellite ECEF positions from SP3 using Lagrange interpolation
- solves receiver ECEF position and clock bias by iterative least squares
- converts to geodetic WGS-84
- estimates velocity by finite differences
- exports CSV and KML

Important limitations of this first version:
- GPS only
- orbit-only SP3 solution; satellite clock handling is simplified
- no precise clock file support yet
- no ionospheric/tropospheric correction yet
- assumes the observation pseudorange is usable directly
- UTC conversion is configurable via a leap-second parameter

Typical use:
    python gnss_offline_path.py \
        --obs gnss_log_2026_03_21_17_17_57.26o \
        --sp3 igr22896.sp3 \
        --out-prefix results/track \
        --gps-utc-leap-seconds 18
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


C = 299_792_458.0
OMEGA_E = 7.2921151467e-5
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


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
        h = p / math.cos(lat) - n
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - n
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

        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
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
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
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

                    if clk_us is not None:
                        current.sat_clocks_s[sat_id] = clk_us * 1e-6

        if not records:
            raise ValueError(f"No SP3 epochs found in {self.path}")
        return records


class Sp3Interpolator:
    """
    Interpolates satellite position and optional SP3 clock values.

    This version uses multi-point Lagrange interpolation instead of the original
    two-point linear interpolation. This is more appropriate for SP3 orbit files,
    where samples are often 5 or 15 minutes apart.

    Recommended values:
    - 9 points: stable first improvement
    - 11 points: good default
    - 13 points: sometimes smoother
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

        # Use local time origin for numerical stability.
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
        """
        Select the closest SP3 records in time that contain the requested satellite.
        """
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
        """
        Barycentric Lagrange interpolation.

        xs: shape (n,)
        ys: shape (n,) or (n, k)
        x: query time coordinate in seconds

        Returns scalar for 1D ys and vector for 2D ys.
        """
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
) -> Optional[Tuple[np.ndarray, float, float, List[str]]]:
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
                sv_pos, sv_clk_s = sp3.interpolate(sat.sat_id, tx_time)
            except KeyError:
                continue

            sv_pos_corr = apply_earth_rotation_correction(sv_pos, transit_time)

            if elevation_mask_deg is not None and np.linalg.norm(rx) > 1e3:
                elev = elevation_angle_deg(rx, sv_pos_corr)
                if elev < elevation_mask_deg:
                    continue

            rho_geom = np.linalg.norm(sv_pos_corr - rx)
            sv_clk_m = C * sv_clk_s if sv_clk_s is not None else 0.0

            pred = rho_geom + cb - sv_clk_m
            v = sat.pseudorange_m - pred

            los = (rx - sv_pos_corr) / rho_geom
            H = np.array([los[0], los[1], los[2], 1.0], dtype=float)

            H_rows.append(H)
            residuals.append(v)
            candidate_sat_ids.append(sat.sat_id)

        if len(H_rows) < 4:
            return None

        H = np.vstack(H_rows)
        y = np.array(residuals, dtype=float)

        try:
            dx, *_ = np.linalg.lstsq(H, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        rx += dx[:3]
        cb += dx[3]
        used_sat_ids = candidate_sat_ids

        if np.linalg.norm(dx[:3]) < convergence_tol_m and abs(dx[3]) < convergence_tol_m:
            break

    final_residuals = []
    final_rows = []
    for sat in sats:
        transit_time = sat.pseudorange_m / C
        tx_time = epoch.time_gps - dt.timedelta(seconds=transit_time)
        try:
            sv_pos, sv_clk_s = sp3.interpolate(sat.sat_id, tx_time)
        except KeyError:
            continue
        sv_pos_corr = apply_earth_rotation_correction(sv_pos, transit_time)

        if elevation_mask_deg is not None:
            elev = elevation_angle_deg(rx, sv_pos_corr)
            if elev < elevation_mask_deg:
                continue

        rho_geom = np.linalg.norm(sv_pos_corr - rx)
        sv_clk_m = C * sv_clk_s if sv_clk_s is not None else 0.0
        pred = rho_geom + cb - sv_clk_m
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
            )

    rms = float(np.sqrt(np.mean(np.square(final_residuals))))
    final_sat_ids = [row[0].sat_id for row in final_rows]
    return rx, cb, rms, final_sat_ids


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


def solve_path(
    obs_path: str,
    sp3_path: str,
    gps_utc_leap_seconds: int,
    min_snr: Optional[float] = None,
    elevation_mask_deg: Optional[float] = None,
    sp3_interp_points: int = 11,
    decimate_to_1hz: bool = True,
) -> List[SolutionEpoch]:
    obs_parser = RinexObsParser(obs_path)
    epochs = obs_parser.parse()
    if decimate_to_1hz:
        epochs = decimate_epochs_to_1hz(epochs)

    sp3_records = Sp3Parser(sp3_path).parse()
    sp3 = Sp3Interpolator(sp3_records, interpolation_points=sp3_interp_points)

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
        )
        if sol is None:
            print(f"[WARN] epoch {idx}/{len(epochs)} {epoch.time_gps.isoformat()} -> no solution", file=sys.stderr)
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline GNSS path solver from RINEX OBS + SP3 precise orbits")
    p.add_argument("--obs", required=True, help="Path to RINEX observation file (.o/.obs)")
    p.add_argument("--sp3", required=True, help="Path to SP3 precise orbit file")
    p.add_argument("--out-prefix", required=True, help="Output prefix, e.g. results/track")
    p.add_argument("--gps-utc-leap-seconds", type=int, default=18,
                   help="GPS-UTC leap seconds to subtract when writing UTC timestamps")
    p.add_argument("--min-snr", type=float, default=None, help="Optional SNR threshold")
    p.add_argument("--elevation-mask-deg", type=float, default=None, help="Optional elevation mask in degrees")
    p.add_argument("--sp3-interp-points", type=int, default=11,
                   help="Number of SP3 epochs for Lagrange interpolation. Try 9, 11, or 13. Default: 11")
    p.add_argument("--no-decimate", action="store_true", help="Do not decimate to 1 Hz")
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
