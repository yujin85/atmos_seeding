#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대관령 ASOS 중심 100×100 WRF 격자의 SEED/NOSEED 강수 및 증우량 산출.

핵심 원칙
---------
1) 한 사례 안에서 SEED와 NOSEED는 동일한 초기시각·runtime을 갖는다고 가정한다.
2) 사례마다 WRF 수행기간, 시딩시간, 시딩 효과시간은 달라도 된다.
3) WRF 파일명/runtime 번호가 아니라 파일 내부의 실제 유효시각(Times)을 사용한다.
4) 시딩 효과시간에 대해 RAINC + RAINNC의 증가량을 누적한다.
5) 대관령 ASOS 최근접 WRF 격자를 중심으로 정확히 100×100 격자를 추출한다.
6) 100×100 영역 평균 증우량 및 논문식 증우율을 계산한다.
7) 같은 시딩 효과시간의 대관령 ASOS 누적강수량을 계산한다.

논문식 증우율
-------------
    P(%) = (SEED - NOSEED) / SEED × 100

관측기반 추정 증우량
-------------------
    ASOS 추정 증우량(mm) = ASOS 누적강수량(mm) × P(%) / 100

필요 패키지
-----------
    pip install numpy pandas xarray netCDF4 matplotlib

실행 전 수정할 곳
----------------
    아래 CASES, ASOS_DIR, OUTPUT_ROOT, 대관령 좌표를 실제 경로/시간으로 수정한다.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# 사용자 설정
# =============================================================================

# 대관령 ASOS
ASOS_STATION_ID = 100
ASOS_STATION_NAME = "Daegwallyeong"
ASOS_LAT = 37.6771
ASOS_LON = 128.7183

# ASOS 시간자료 디렉터리: KMA_ASOS_YYYYMMDDHH.dat 파일이 여러 날짜에 걸쳐 존재
ASOS_DIR = Path("/path/to/ASOS")
ASOS_GLOB = "KMA_ASOS_*.dat"

# 결과 최상위 디렉터리
OUTPUT_ROOT = Path("./WRF_ASOS_PRECIP_ENHANCEMENT")

# 추출 격자 수
SUBDOMAIN_NY = 100
SUBDOMAIN_NX = 100

# WRF 파일 검색 패턴
WRF_GLOB = "wrfout_d0*"

# 효과시간 경계에 정확한 wrfout 시각이 없을 때 허용할 최근접 시각 오차
# 정확 일치만 허용하려면 0으로 설정
WRF_TIME_TOLERANCE_MINUTES = 0

# ASOS 시각 허용 오차. 일반적인 시간자료는 0 권장
ASOS_TIME_TOLERANCE_MINUTES = 0

# 논문 그림에서 사용한 증우 판정 최소값 예: 0.01 mm
ENHANCEMENT_THRESHOLD_MM = 0.01

# SEED 평균강수량이 이 값 이하이면 증우율을 NaN 처리
MIN_SEED_MEAN_FOR_RATE_MM = 1.0e-12

# WRF 중복 유효시각 처리 정책: "error", "last", "first"
DUPLICATE_WRF_TIME_POLICY = "error"

# 그림 저장 여부
SAVE_FIGURES = True

# -----------------------------------------------------------------------------
# 사례별 설정
# 모든 시각은 KST로 입력한다.
# WRF 파일 내부 Times는 일반적으로 UTC이므로 wrf_time_basis="UTC"를 사용한다.
# 만약 파일 내부 Times가 KST라면 "KST"로 변경한다.
# -----------------------------------------------------------------------------
CASES = [
    {
        "case_name": "case_20250916",
        "noseed_dir": "/path/to/case_20250916/NOSEED",
        "seed_dir": "/path/to/case_20250916/SEED",
        "wrf_time_basis": "UTC",  # "UTC" 또는 "KST"

        # 실제 시딩 수행시간: 기록·출력용
        "seeding_start_kst": "2025-09-16 09:30:00",
        "seeding_end_kst": "2025-09-16 10:00:00",

        # 실제 강수/증우량을 계산할 시딩 효과시간
        "effect_start_kst": "2025-09-16 10:00:00",
        "effect_end_kst": "2025-09-16 19:00:00",
    },
]


# =============================================================================
# 자료 구조
# =============================================================================

@dataclass(frozen=True)
class WRFTimeRecord:
    valid_time_kst: datetime
    valid_time_native: datetime
    file_path: Path
    time_index: int


# =============================================================================
# 공통 유틸리티
# =============================================================================

def parse_datetime(value: str) -> datetime:
    """여러 일반적인 문자열 형식을 datetime으로 변환한다."""
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M",
        "%Y%m%d%H",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"지원하지 않는 날짜 형식입니다: {value}")


def native_to_kst(value: datetime, basis: str) -> datetime:
    basis_upper = basis.upper()
    if basis_upper == "UTC":
        return value + timedelta(hours=9)
    if basis_upper == "KST":
        return value
    raise ValueError(f"wrf_time_basis는 UTC 또는 KST여야 합니다: {basis}")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_case(case: dict) -> None:
    required = {
        "case_name",
        "noseed_dir",
        "seed_dir",
        "wrf_time_basis",
        "seeding_start_kst",
        "seeding_end_kst",
        "effect_start_kst",
        "effect_end_kst",
    }
    missing = required.difference(case)
    if missing:
        raise KeyError(f"사례 설정 누락: {sorted(missing)}")

    seed_start = parse_datetime(case["seeding_start_kst"])
    seed_end = parse_datetime(case["seeding_end_kst"])
    effect_start = parse_datetime(case["effect_start_kst"])
    effect_end = parse_datetime(case["effect_end_kst"])

    if seed_end <= seed_start:
        raise ValueError(f"{case['case_name']}: 시딩 종료가 시작보다 늦어야 합니다.")
    if effect_end <= effect_start:
        raise ValueError(f"{case['case_name']}: 효과 종료가 시작보다 늦어야 합니다.")


def clean_netcdf_attr(value: object) -> object:
    """NetCDF attribute로 안전하게 저장할 수 있는 값으로 변환한다."""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN"
    return value


# =============================================================================
# WRF 시간 및 변수 읽기
# =============================================================================

def decode_wrf_times(ds: xr.Dataset) -> List[datetime]:
    """WRF Times 또는 datetime64형 Time 좌표를 실제 datetime 목록으로 읽는다."""
    if "Times" in ds.variables:
        raw = ds["Times"].values

        if raw.ndim == 2:
            strings: List[str] = []
            for row in raw:
                if row.dtype.kind in {"S", "a"}:
                    text = b"".join(row.tolist()).decode("ascii", errors="ignore")
                else:
                    text = "".join(
                        x.decode("ascii", errors="ignore") if isinstance(x, bytes) else str(x)
                        for x in row.tolist()
                    )
                strings.append(text.strip().replace("\x00", ""))
        elif raw.ndim == 1:
            strings = []
            for item in raw.tolist():
                if isinstance(item, bytes):
                    strings.append(item.decode("ascii", errors="ignore").strip())
                else:
                    strings.append(str(item).strip())
        else:
            raise ValueError(f"예상하지 못한 Times 차원: {raw.shape}")

        result = []
        for text in strings:
            parsed = None
            for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    pass
            if parsed is None:
                raise ValueError(f"WRF Times 해석 실패: {text!r}")
            result.append(parsed)
        return result

    # 일부 변환 NetCDF는 Time 좌표가 datetime64일 수 있음
    if "Time" in ds.coords and np.issubdtype(ds["Time"].dtype, np.datetime64):
        return [pd.Timestamp(v).to_pydatetime() for v in ds["Time"].values]

    raise KeyError("WRF 파일에 Times 또는 datetime형 Time 좌표가 없습니다.")


def build_wrf_time_index(directory: Path, time_basis: str) -> Dict[datetime, WRFTimeRecord]:
    """디렉터리 전체 wrfout의 모든 Time을 KST 기준 색인한다."""
    if DUPLICATE_WRF_TIME_POLICY not in {"error", "last", "first"}:
        raise ValueError(
            'DUPLICATE_WRF_TIME_POLICY는 "error", "last", "first" 중 하나여야 합니다.'
        )
    if not directory.exists():
        raise FileNotFoundError(f"WRF 디렉터리가 없습니다: {directory}")

    files = sorted(p for p in directory.glob(WRF_GLOB) if p.is_file())
    if not files:
        raise FileNotFoundError(f"WRF 파일을 찾지 못했습니다: {directory}/{WRF_GLOB}")

    index: Dict[datetime, WRFTimeRecord] = {}

    for path in files:
        try:
            with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
                native_times = decode_wrf_times(ds)
        except Exception as exc:
            raise RuntimeError(f"WRF 시간 읽기 실패: {path}\n{exc}") from exc

        for time_index, native_time in enumerate(native_times):
            kst_time = native_to_kst(native_time, time_basis)
            record = WRFTimeRecord(kst_time, native_time, path, time_index)

            if kst_time in index:
                previous = index[kst_time]
                message = (
                    f"중복 WRF 유효시각 {kst_time}:\n"
                    f"  기존: {previous.file_path} Time={previous.time_index}\n"
                    f"  신규: {path} Time={time_index}"
                )
                if DUPLICATE_WRF_TIME_POLICY == "error":
                    raise ValueError(message)
                if DUPLICATE_WRF_TIME_POLICY == "first":
                    warnings.warn(message + "\n기존 레코드를 사용합니다.")
                    continue
                warnings.warn(message + "\n신규 레코드를 사용합니다.")
            index[kst_time] = record

    return dict(sorted(index.items()))


def select_time_record(
    index: Dict[datetime, WRFTimeRecord],
    target: datetime,
    tolerance_minutes: int,
    label: str,
) -> WRFTimeRecord:
    """정확 시각 또는 허용오차 내 최근접 WRF 레코드를 선택한다."""
    if target in index:
        return index[target]

    if tolerance_minutes <= 0:
        available_min = min(index)
        available_max = max(index)
        raise KeyError(
            f"{label}: 정확한 WRF 시각이 없습니다: {target} KST\n"
            f"가용 범위: {available_min} ~ {available_max} KST"
        )

    nearest_time = min(index, key=lambda x: abs(x - target))
    delta_minutes = abs((nearest_time - target).total_seconds()) / 60.0
    if delta_minutes > tolerance_minutes:
        raise KeyError(
            f"{label}: 목표시각 {target} KST와 최근접 WRF 시각 {nearest_time} KST의 "
            f"차이 {delta_minutes:.1f}분이 허용오차 {tolerance_minutes}분보다 큽니다."
        )

    warnings.warn(
        f"{label}: 목표시각 {target} 대신 최근접 WRF 시각 {nearest_time} KST 사용 "
        f"({delta_minutes:.1f}분 차이)"
    )
    return index[nearest_time]


def read_2d_variable(record: WRFTimeRecord, variable_name: str) -> np.ndarray:
    """지정 레코드에서 2차원 변수를 읽는다."""
    with xr.open_dataset(record.file_path, decode_times=False, mask_and_scale=True) as ds:
        if variable_name not in ds.variables:
            raise KeyError(f"{record.file_path}에 {variable_name} 변수가 없습니다.")
        var = ds[variable_name]
        if "Time" in var.dims:
            var = var.isel(Time=record.time_index)
        array = np.asarray(var.values, dtype=np.float64)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(
                f"{variable_name}가 2차원이 아닙니다: {record.file_path}, shape={array.shape}"
            )
        return array


def read_total_accumulated_rain(record: WRFTimeRecord) -> np.ndarray:
    """RAINC + RAINNC 누적강수량(mm)을 읽는다."""
    return read_2d_variable(record, "RAINC") + read_2d_variable(record, "RAINNC")


def read_lat_lon(record: WRFTimeRecord) -> Tuple[np.ndarray, np.ndarray]:
    """XLAT/XLONG을 읽는다."""
    lat = read_2d_variable(record, "XLAT")
    lon = read_2d_variable(record, "XLONG")
    return lat, lon


def get_records_in_period(
    index: Dict[datetime, WRFTimeRecord],
    start: datetime,
    end: datetime,
    tolerance_minutes: int,
    label: str,
) -> List[WRFTimeRecord]:
    """효과 시작·종료 경계를 포함하는 연속 WRF 레코드 목록을 반환한다."""
    start_record = select_time_record(index, start, tolerance_minutes, f"{label} 시작")
    end_record = select_time_record(index, end, tolerance_minutes, f"{label} 종료")

    actual_start = start_record.valid_time_kst
    actual_end = end_record.valid_time_kst
    if actual_end <= actual_start:
        raise ValueError(f"{label}: 효과 종료시각이 시작시각보다 늦어야 합니다.")

    records = [record for t, record in index.items() if actual_start <= t <= actual_end]
    if len(records) < 2:
        raise ValueError(f"{label}: 기간강수 계산에 최소 2개 WRF 시각이 필요합니다.")
    return records


def calculate_period_precipitation(
    records: Sequence[WRFTimeRecord],
    label: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    효과시간 동안 누적강수 증가량을 합산한다.

    일반적인 경우 마지막 누적값 - 첫 누적값과 같다.
    WRF restart로 RAINC/RAINNC가 초기화되어 음의 차이가 생기면,
    해당 구간의 현재 누적값을 증가량으로 간주하여 이어 붙인다.
    """
    previous = read_total_accumulated_rain(records[0])
    period_total = np.zeros_like(previous, dtype=np.float64)
    rows = []

    for previous_record, current_record in zip(records[:-1], records[1:]):
        current = read_total_accumulated_rain(current_record)
        if current.shape != previous.shape:
            raise ValueError(f"{label}: WRF 격자 크기가 시간에 따라 달라졌습니다.")

        raw_increment = current - previous
        reset_mask = raw_increment < -1.0e-6
        tiny_negative_mask = (raw_increment < 0.0) & ~reset_mask

        # restart/reset 시 현재 누적값을 그 구간의 증가량으로 사용
        increment = np.where(reset_mask, current, raw_increment)
        increment = np.where(tiny_negative_mask, 0.0, increment)
        increment = np.where(np.isfinite(increment), increment, np.nan)

        period_total += np.nan_to_num(increment, nan=0.0)

        rows.append(
            {
                "interval_start_kst": previous_record.valid_time_kst,
                "interval_end_kst": current_record.valid_time_kst,
                "domain_increment_mean_mm": float(np.nanmean(increment)),
                "domain_increment_max_mm": float(np.nanmax(increment)),
                "reset_grid_count": int(np.count_nonzero(reset_mask)),
            }
        )

        previous = current

    return period_total, pd.DataFrame(rows)


def validate_seed_noseed_times(
    seed_records: Sequence[WRFTimeRecord],
    noseed_records: Sequence[WRFTimeRecord],
) -> None:
    seed_times = [r.valid_time_kst for r in seed_records]
    noseed_times = [r.valid_time_kst for r in noseed_records]
    if seed_times != noseed_times:
        only_seed = sorted(set(seed_times).difference(noseed_times))
        only_noseed = sorted(set(noseed_times).difference(seed_times))
        raise ValueError(
            "SEED와 NOSEED의 효과시간 WRF 유효시각 목록이 일치하지 않습니다.\n"
            f"SEED에만 존재: {only_seed[:10]}\n"
            f"NOSEED에만 존재: {only_noseed[:10]}"
        )


def validate_seed_noseed_grid(
    seed_record: WRFTimeRecord,
    noseed_record: WRFTimeRecord,
    case_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """SEED/NOSEED의 위경도 격자가 같은지 확인하고 NOSEED 위경도를 반환한다."""
    noseed_lat, noseed_lon = read_lat_lon(noseed_record)
    seed_lat, seed_lon = read_lat_lon(seed_record)

    if noseed_lat.shape != seed_lat.shape or noseed_lon.shape != seed_lon.shape:
        raise ValueError(f"{case_name}: SEED/NOSEED XLAT/XLONG shape가 다릅니다.")

    lat_same = np.allclose(noseed_lat, seed_lat, rtol=0.0, atol=1.0e-8, equal_nan=True)
    lon_same = np.allclose(noseed_lon, seed_lon, rtol=0.0, atol=1.0e-8, equal_nan=True)
    if not lat_same or not lon_same:
        raise ValueError(f"{case_name}: SEED/NOSEED XLAT/XLONG이 일치하지 않습니다.")

    return noseed_lat, noseed_lon


# =============================================================================
# 100×100 영역
# =============================================================================

def find_nearest_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> Tuple[int, int]:
    """구면거리 근사로 최근접 격자를 찾는다."""
    lon_weight = math.cos(math.radians(target_lat))
    distance2 = (lat2d - target_lat) ** 2 + ((lon2d - target_lon) * lon_weight) ** 2
    j, i = np.unravel_index(np.nanargmin(distance2), distance2.shape)
    return int(j), int(i)


def make_fixed_subdomain_slices(
    center_j: int,
    center_i: int,
    domain_ny: int,
    domain_nx: int,
    sub_ny: int,
    sub_nx: int,
) -> Tuple[slice, slice]:
    """도메인 경계에서도 가능한 한 정확한 sub_ny×sub_nx 크기를 유지한다."""
    if sub_ny > domain_ny or sub_nx > domain_nx:
        raise ValueError(
            f"요청 영역 {sub_ny}×{sub_nx}가 WRF 도메인 {domain_ny}×{domain_nx}보다 큽니다."
        )

    j0 = center_j - sub_ny // 2
    i0 = center_i - sub_nx // 2
    j0 = max(0, min(j0, domain_ny - sub_ny))
    i0 = max(0, min(i0, domain_nx - sub_nx))
    return slice(j0, j0 + sub_ny), slice(i0, i0 + sub_nx)


def read_grid_area(record: WRFTimeRecord, shape: Tuple[int, int]) -> np.ndarray:
    """격자면적(m²)을 계산한다. MAPFAC 계열 변수가 있으면 투영 보정을 적용한다."""
    with xr.open_dataset(record.file_path, decode_times=False, mask_and_scale=True) as ds:
        if "DX" not in ds.attrs or "DY" not in ds.attrs:
            raise KeyError(f"{record.file_path}의 전역속성에 DX 또는 DY가 없습니다.")
        dx = float(ds.attrs["DX"])
        dy = float(ds.attrs["DY"])

        if "MAPFAC_M" in ds.variables:
            mapfac = ds["MAPFAC_M"]
            if "Time" in mapfac.dims:
                mapfac = mapfac.isel(Time=record.time_index)
            mapfac_array = np.squeeze(np.asarray(mapfac.values, dtype=np.float64))
            if mapfac_array.shape != shape:
                raise ValueError(
                    f"MAPFAC_M shape {mapfac_array.shape}와 강수 shape {shape}가 다릅니다."
                )
            return dx * dy / np.square(mapfac_array)

        if "MAPFAC_MX" in ds.variables and "MAPFAC_MY" in ds.variables:
            mapfac_x = ds["MAPFAC_MX"]
            mapfac_y = ds["MAPFAC_MY"]
            if "Time" in mapfac_x.dims:
                mapfac_x = mapfac_x.isel(Time=record.time_index)
            if "Time" in mapfac_y.dims:
                mapfac_y = mapfac_y.isel(Time=record.time_index)
            mapfac_x_array = np.squeeze(np.asarray(mapfac_x.values, dtype=np.float64))
            mapfac_y_array = np.squeeze(np.asarray(mapfac_y.values, dtype=np.float64))
            if mapfac_x_array.shape != shape or mapfac_y_array.shape != shape:
                raise ValueError("MAPFAC_MX/MY shape와 강수 shape가 다릅니다.")
            return dx * dy / (mapfac_x_array * mapfac_y_array)

        return np.full(shape, dx * dy, dtype=np.float64)


# =============================================================================
# ASOS 읽기
# =============================================================================

def parse_asos_data_line(line: str) -> Optional[dict]:
    """
    기상청 ASOS help=1 텍스트 한 줄을 파싱한다.

    WW와 CT가 문자열이고 공백 폭이 고정되어 있어 단순 split만으로는 항상 전체 열이
    보장되지 않는다. 그러나 TM/STN/RN/RN_DAY는 앞쪽 고정 순서이며 split 결과의
    첫 17개 토큰으로 안정적으로 읽을 수 있다.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens = stripped.split()
    if len(tokens) < 17:
        return None

    try:
        tm = datetime.strptime(tokens[0], "%Y%m%d%H%M")
        stn = int(tokens[1])
        rn = float(tokens[15])
        rn_day = float(tokens[16])
    except (ValueError, IndexError):
        return None

    return {"TM": tm, "STN": stn, "RN": rn, "RN_DAY": rn_day}


def read_asos_directory(directory: Path, station_id: int) -> pd.DataFrame:
    """디렉터리 전체 ASOS 파일에서 특정 지점 자료를 읽는다."""
    if not directory.exists():
        raise FileNotFoundError(f"ASOS 디렉터리가 없습니다: {directory}")

    files = sorted(p for p in directory.glob(ASOS_GLOB) if p.is_file())
    if not files:
        raise FileNotFoundError(f"ASOS 파일을 찾지 못했습니다: {directory}/{ASOS_GLOB}")

    rows = []
    for path in files:
        with path.open("r", encoding="euc-kr", errors="replace") as f:
            for line in f:
                row = parse_asos_data_line(line)
                if row is not None and row["STN"] == station_id:
                    row["source_file"] = str(path)
                    rows.append(row)

    if not rows:
        raise ValueError(f"ASOS STN={station_id} 자료를 찾지 못했습니다: {directory}")

    df = pd.DataFrame(rows)
    df = df.sort_values("TM").drop_duplicates(subset=["TM"], keep="last").reset_index(drop=True)

    # 결측코드 처리
    for column in ("RN", "RN_DAY"):
        df.loc[df[column] <= -8.9, column] = np.nan

    return df


def nearest_asos_time(
    available_times: Sequence[datetime],
    target: datetime,
    tolerance_minutes: int,
    label: str,
) -> datetime:
    available_time_set = set(available_times)
    if target in available_time_set:
        return target
    if tolerance_minutes <= 0:
        raise KeyError(f"{label}: ASOS 시각 {target} KST가 없습니다.")

    nearest = min(available_times, key=lambda x: abs(x - target))
    delta = abs((nearest - target).total_seconds()) / 60.0
    if delta > tolerance_minutes:
        raise KeyError(
            f"{label}: 목표시각과 최근접 ASOS 시각 차이 {delta:.1f}분이 "
            f"허용오차 {tolerance_minutes}분보다 큽니다."
        )
    warnings.warn(f"{label}: ASOS 목표시각 {target} 대신 {nearest} 사용")
    return nearest


def calculate_asos_period_precipitation(
    asos_df: pd.DataFrame,
    start_kst: datetime,
    end_kst: datetime,
    tolerance_minutes: int,
) -> Tuple[float, pd.DataFrame, str]:
    """
    ASOS 효과시간 누적강수량을 계산한다.

    우선 RN_DAY 누적값의 시간차분을 사용한다. 이는 RN이 계절에 따라 1시간 또는
    3시간 강수량인 문제를 피하기 위함이다. 날짜가 바뀌면 RN_DAY가 초기화되므로
    새 날짜의 첫 값 자체를 자정 이후 누적 증가량으로 사용한다.

    시간구간은 (start, end]로 계산한다. 즉 10~11시 강수는 11시 자료에 포함된다.
    시작시각 자료가 있어야 첫 차분을 정확히 계산할 수 있다.
    """
    available_times = asos_df["TM"].tolist()
    actual_start = nearest_asos_time(available_times, start_kst, tolerance_minutes, "ASOS 효과 시작")
    actual_end = nearest_asos_time(available_times, end_kst, tolerance_minutes, "ASOS 효과 종료")

    if actual_end <= actual_start:
        raise ValueError("ASOS 효과 종료시각이 시작시각보다 늦어야 합니다.")

    start_rows = asos_df.index[asos_df["TM"] == actual_start].tolist()
    end_rows = asos_df.index[asos_df["TM"] == actual_end].tolist()
    if not start_rows or not end_rows:
        raise KeyError("ASOS 효과시간 경계 자료를 찾지 못했습니다.")

    start_idx = start_rows[-1]
    end_idx = end_rows[-1]
    period = asos_df.loc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(period) < 2:
        raise ValueError("ASOS 기간강수 계산에 최소 2개 시각이 필요합니다.")

    increments = [np.nan]
    methods = ["boundary"]

    for idx in range(1, len(period)):
        prev = period.iloc[idx - 1]
        curr = period.iloc[idx]
        increment = np.nan
        method = "missing"

        prev_day = prev["TM"].date()
        curr_day = curr["TM"].date()
        prev_cum = prev["RN_DAY"]
        curr_cum = curr["RN_DAY"]

        if pd.notna(curr_cum):
            if curr_day == prev_day and pd.notna(prev_cum):
                diff = float(curr_cum - prev_cum)
                if diff >= -1.0e-6:
                    increment = max(diff, 0.0)
                    method = "RN_DAY_diff"
                elif pd.notna(curr["RN"]) and curr["RN"] >= 0.0:
                    # 같은 날짜인데 누적값이 감소하면 품질 문제로 RN 대체 시도
                    increment = float(curr["RN"])
                    method = "RN_fallback_after_RN_DAY_drop"
            else:
                # 날짜가 바뀌었으면 현재 일누적 자체가 자정 이후 증가량
                increment = max(float(curr_cum), 0.0)
                method = "RN_DAY_new_day"

        if pd.isna(increment) and pd.notna(curr["RN"]) and curr["RN"] >= 0.0:
            # RN은 여름철 1시간, 겨울철 3시간일 수 있으므로 보조수단으로만 사용
            increment = float(curr["RN"])
            method = "RN_fallback"

        increments.append(increment)
        methods.append(method)

    period["interval_precip_mm"] = increments
    period["precip_method"] = methods

    missing = period.loc[1:, "interval_precip_mm"].isna()
    if missing.any():
        bad_times = period.loc[1:][missing]["TM"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        raise ValueError(f"ASOS 강수 증가량을 계산할 수 없는 시각: {bad_times}")

    total = float(period.loc[1:, "interval_precip_mm"].sum())
    method_summary = ",".join(sorted(set(period.loc[1:, "precip_method"])))
    return total, period, method_summary


# =============================================================================
# 통계·저장·그림
# =============================================================================

def safe_nanmin(array: np.ndarray) -> float:
    return float(np.nanmin(array)) if np.isfinite(array).any() else float("nan")


def safe_nanmax(array: np.ndarray) -> float:
    return float(np.nanmax(array)) if np.isfinite(array).any() else float("nan")


def save_maps(
    output_dir: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    noseed: np.ndarray,
    seed: np.ndarray,
    enhancement: np.ndarray,
    center_lon: float,
    center_lat: float,
    case_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("그림 저장에는 matplotlib이 필요합니다. pip install matplotlib 또는 --no-figures를 사용하세요.") from exc

    figures = [
        (noseed, "NOSEED accumulated precipitation", "mm", "noseed_precip.png", None),
        (seed, "SEED accumulated precipitation", "mm", "seed_precip.png", None),
        (enhancement, "Precipitation enhancement (SEED - NOSEED)", "mm", "enhancement.png", "RdBu_r"),
    ]

    for values, title, units, filename, cmap in figures:
        fig, ax = plt.subplots(figsize=(9, 8))
        kwargs = {"shading": "auto"}
        if cmap is not None:
            kwargs["cmap"] = cmap
        mesh = ax.pcolormesh(lon, lat, values, **kwargs)
        ax.scatter([center_lon], [center_lat], marker="*", s=100, label=ASOS_STATION_NAME)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{case_name}: {title}")
        ax.legend(loc="best")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(units)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_netcdf(
    output_path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    grid_area: np.ndarray,
    noseed: np.ndarray,
    seed: np.ndarray,
    enhancement: np.ndarray,
    enhancement_rate_grid: np.ndarray,
    j_indices: np.ndarray,
    i_indices: np.ndarray,
    attrs: dict,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "NOSEED_PRECIP": (
                ("south_north", "west_east"), noseed,
                {"long_name": "NOSEED accumulated precipitation during effect period", "units": "mm"},
            ),
            "SEED_PRECIP": (
                ("south_north", "west_east"), seed,
                {"long_name": "SEED accumulated precipitation during effect period", "units": "mm"},
            ),
            "PRECIP_ENHANCEMENT": (
                ("south_north", "west_east"), enhancement,
                {"long_name": "SEED minus NOSEED precipitation", "units": "mm"},
            ),
            "ENHANCEMENT_RATE_GRID": (
                ("south_north", "west_east"), enhancement_rate_grid,
                {"long_name": "Grid precipitation enhancement rate relative to SEED", "units": "%"},
            ),
            "GRID_AREA": (
                ("south_north", "west_east"), grid_area,
                {"long_name": "WRF grid-cell area", "units": "m2"},
            ),
            "XLAT": (
                ("south_north", "west_east"), lat,
                {"long_name": "latitude", "units": "degrees_north"},
            ),
            "XLONG": (
                ("south_north", "west_east"), lon,
                {"long_name": "longitude", "units": "degrees_east"},
            ),
        },
        coords={
            "south_north": j_indices,
            "west_east": i_indices,
        },
        attrs=attrs,
    )

    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)


def process_case(case: dict, asos_df: pd.DataFrame) -> dict:
    validate_case(case)

    case_name = case["case_name"]
    output_dir = OUTPUT_ROOT / case_name
    ensure_directory(output_dir)

    seeding_start = parse_datetime(case["seeding_start_kst"])
    seeding_end = parse_datetime(case["seeding_end_kst"])
    effect_start = parse_datetime(case["effect_start_kst"])
    effect_end = parse_datetime(case["effect_end_kst"])

    noseed_dir = Path(case["noseed_dir"])
    seed_dir = Path(case["seed_dir"])
    time_basis = case["wrf_time_basis"]

    print("\n" + "=" * 90)
    print(f"CASE: {case_name}")
    print("=" * 90)
    print(f"Seeding KST : {seeding_start} ~ {seeding_end}")
    print(f"Effect  KST : {effect_start} ~ {effect_end}")

    print("[1/8] WRF 시간 색인 생성")
    noseed_index = build_wrf_time_index(noseed_dir, time_basis)
    seed_index = build_wrf_time_index(seed_dir, time_basis)

    noseed_records = get_records_in_period(
        noseed_index, effect_start, effect_end, WRF_TIME_TOLERANCE_MINUTES, "NOSEED"
    )
    seed_records = get_records_in_period(
        seed_index, effect_start, effect_end, WRF_TIME_TOLERANCE_MINUTES, "SEED"
    )
    validate_seed_noseed_times(seed_records, noseed_records)

    actual_effect_start = seed_records[0].valid_time_kst
    actual_effect_end = seed_records[-1].valid_time_kst

    print("[2/8] WRF 기간 누적강수량 계산")
    noseed_full, noseed_intervals = calculate_period_precipitation(noseed_records, "NOSEED")
    seed_full, seed_intervals = calculate_period_precipitation(seed_records, "SEED")

    if noseed_full.shape != seed_full.shape:
        raise ValueError(
            f"{case_name}: SEED/NOSEED 격자 크기가 다릅니다: "
            f"{seed_full.shape} != {noseed_full.shape}"
        )

    print("[3/8] 대관령 최근접 격자 및 100×100 영역 추출")
    lat_full, lon_full = validate_seed_noseed_grid(seed_records[0], noseed_records[0], case_name)
    center_j, center_i = find_nearest_grid(lat_full, lon_full, ASOS_LAT, ASOS_LON)
    j_slice, i_slice = make_fixed_subdomain_slices(
        center_j, center_i,
        noseed_full.shape[0], noseed_full.shape[1],
        SUBDOMAIN_NY, SUBDOMAIN_NX,
    )

    lat = lat_full[j_slice, i_slice]
    lon = lon_full[j_slice, i_slice]
    noseed = noseed_full[j_slice, i_slice]
    seed = seed_full[j_slice, i_slice]
    enhancement = seed - noseed

    grid_area_full = read_grid_area(noseed_records[0], noseed_full.shape)
    grid_area = grid_area_full[j_slice, i_slice]

    valid = np.isfinite(seed) & np.isfinite(noseed) & np.isfinite(grid_area) & (grid_area > 0)
    if not np.any(valid):
        raise ValueError(f"{case_name}: 100×100 영역에 유효 격자가 없습니다.")

    print("[4/8] 격자별·100×100 영역 통계 계산")
    seed_mean = float(np.nanmean(np.where(valid, seed, np.nan)))
    noseed_mean = float(np.nanmean(np.where(valid, noseed, np.nan)))
    enhancement_mean = float(np.nanmean(np.where(valid, enhancement, np.nan)))

    if seed_mean > MIN_SEED_MEAN_FOR_RATE_MM:
        enhancement_rate_100x100 = enhancement_mean / seed_mean * 100.0
    else:
        enhancement_rate_100x100 = float("nan")

    enhancement_rate_grid = np.where(
        valid & (seed > MIN_SEED_MEAN_FOR_RATE_MM),
        enhancement / seed * 100.0,
        np.nan,
    )

    station_seed = float(seed_full[center_j, center_i])
    station_noseed = float(noseed_full[center_j, center_i])
    station_enhancement = station_seed - station_noseed
    if np.isfinite(station_seed) and station_seed > MIN_SEED_MEAN_FOR_RATE_MM:
        station_enhancement_rate = station_enhancement / station_seed * 100.0
    else:
        station_enhancement_rate = float("nan")

    positive_mask = valid & (enhancement > ENHANCEMENT_THRESHOLD_MM)
    negative_mask = valid & (enhancement < -ENHANCEMENT_THRESHOLD_MM)
    neutral_mask = valid & ~(positive_mask | negative_mask)

    positive_grid_count = int(np.count_nonzero(positive_mask))
    negative_grid_count = int(np.count_nonzero(negative_mask))
    neutral_grid_count = int(np.count_nonzero(neutral_mask))
    valid_grid_count = int(np.count_nonzero(valid))

    positive_area_km2 = float(np.sum(grid_area[positive_mask]) / 1.0e6)
    negative_area_km2 = float(np.sum(grid_area[negative_mask]) / 1.0e6)
    analysis_area_km2 = float(np.sum(grid_area[valid]) / 1.0e6)

    # mm × m² × 0.001 = m³; 물 1m³ ≈ 1톤
    net_enhanced_volume_m3 = float(np.sum(enhancement[valid] * grid_area[valid] * 0.001))
    positive_volume_m3 = float(np.sum(enhancement[positive_mask] * grid_area[positive_mask] * 0.001))
    negative_volume_m3 = float(np.sum(enhancement[negative_mask] * grid_area[negative_mask] * 0.001))

    positive_grid_fraction_percent = positive_grid_count / valid_grid_count * 100.0
    seed_total_reset_grid_count = int(seed_intervals["reset_grid_count"].sum())
    noseed_total_reset_grid_count = int(noseed_intervals["reset_grid_count"].sum())

    print("[5/8] 같은 효과시간의 대관령 ASOS 누적강수량 계산")
    asos_precip, asos_period, asos_method = calculate_asos_period_precipitation(
        asos_df, actual_effect_start, actual_effect_end, ASOS_TIME_TOLERANCE_MINUTES
    )

    if np.isfinite(enhancement_rate_100x100):
        asos_estimated_enhancement = asos_precip * enhancement_rate_100x100 / 100.0
    else:
        asos_estimated_enhancement = float("nan")

    if np.isfinite(station_enhancement_rate):
        asos_estimated_enhancement_station_rate = asos_precip * station_enhancement_rate / 100.0
    else:
        asos_estimated_enhancement_station_rate = float("nan")

    print("[6/8] CSV 저장")
    summary = {
        "case_name": case_name,
        "station_id": ASOS_STATION_ID,
        "station_name": ASOS_STATION_NAME,
        "station_lat": ASOS_LAT,
        "station_lon": ASOS_LON,
        "nearest_wrf_j": center_j,
        "nearest_wrf_i": center_i,
        "nearest_wrf_lat": float(lat_full[center_j, center_i]),
        "nearest_wrf_lon": float(lon_full[center_j, center_i]),
        "subdomain_ny": SUBDOMAIN_NY,
        "subdomain_nx": SUBDOMAIN_NX,
        "subdomain_j_start": j_slice.start,
        "subdomain_j_end_inclusive": j_slice.stop - 1,
        "subdomain_i_start": i_slice.start,
        "subdomain_i_end_inclusive": i_slice.stop - 1,
        "seeding_start_kst": seeding_start,
        "seeding_end_kst": seeding_end,
        "requested_effect_start_kst": effect_start,
        "requested_effect_end_kst": effect_end,
        "actual_effect_start_kst": actual_effect_start,
        "actual_effect_end_kst": actual_effect_end,
        "wrf_time_count": len(seed_records),
        "seed_total_reset_grid_count": seed_total_reset_grid_count,
        "noseed_total_reset_grid_count": noseed_total_reset_grid_count,
        "analysis_area_km2": analysis_area_km2,
        "seed_mean_100x100_mm": seed_mean,
        "noseed_mean_100x100_mm": noseed_mean,
        "mean_enhancement_100x100_mm": enhancement_mean,
        "enhancement_rate_100x100_percent": enhancement_rate_100x100,
        "station_grid_seed_precip_mm": station_seed,
        "station_grid_noseed_precip_mm": station_noseed,
        "station_grid_enhancement_mm": station_enhancement,
        "station_grid_enhancement_rate_percent": station_enhancement_rate,
        "grid_enhancement_max_mm": safe_nanmax(enhancement),
        "grid_enhancement_min_mm": safe_nanmin(enhancement),
        "enhancement_threshold_mm": ENHANCEMENT_THRESHOLD_MM,
        "valid_grid_count": valid_grid_count,
        "positive_grid_count": positive_grid_count,
        "negative_grid_count": negative_grid_count,
        "neutral_grid_count": neutral_grid_count,
        "positive_grid_fraction_percent": positive_grid_fraction_percent,
        "positive_area_km2": positive_area_km2,
        "negative_area_km2": negative_area_km2,
        "positive_enhanced_volume_m3_ton": positive_volume_m3,
        "negative_enhanced_volume_m3_ton": negative_volume_m3,
        "net_enhanced_volume_m3_ton": net_enhanced_volume_m3,
        "asos_accumulated_precip_mm": asos_precip,
        "asos_precip_method": asos_method,
        "asos_estimated_enhancement_mm": asos_estimated_enhancement,
        "asos_estimated_enhancement_station_rate_mm": asos_estimated_enhancement_station_rate,
        "enhancement_formula": "(SEED-NOSEED)/SEED*100",
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )

    j_indices = np.arange(j_slice.start, j_slice.stop)
    i_indices = np.arange(i_slice.start, i_slice.stop)
    grid_df = pd.DataFrame(
        {
            "wrf_j": np.repeat(j_indices, len(i_indices)),
            "wrf_i": np.tile(i_indices, len(j_indices)),
            "latitude": lat.ravel(),
            "longitude": lon.ravel(),
            "grid_area_m2": grid_area.ravel(),
            "noseed_precip_mm": noseed.ravel(),
            "seed_precip_mm": seed.ravel(),
            "enhancement_mm": enhancement.ravel(),
            "enhancement_rate_percent": enhancement_rate_grid.ravel(),
            "enhancement_class": np.where(
                positive_mask.ravel(), "positive",
                np.where(negative_mask.ravel(), "negative", "neutral"),
            ),
        }
    )
    grid_df.to_csv(output_dir / "grid_100x100.csv", index=False, encoding="utf-8-sig")

    seed_intervals.to_csv(output_dir / "seed_time_intervals.csv", index=False, encoding="utf-8-sig")
    noseed_intervals.to_csv(output_dir / "noseed_time_intervals.csv", index=False, encoding="utf-8-sig")
    asos_period.to_csv(output_dir / "asos_period.csv", index=False, encoding="utf-8-sig")

    print("[7/8] NetCDF 저장")
    netcdf_attrs = {
        key: clean_netcdf_attr(value)
        for key, value in summary.items()
        if value is not None
    }
    save_netcdf(
        output_dir / "precip_enhancement_100x100.nc",
        lat, lon, grid_area, noseed, seed, enhancement, enhancement_rate_grid,
        j_indices, i_indices, netcdf_attrs,
    )

    if SAVE_FIGURES:
        print("[8/8] 분포 그림 저장")
        save_maps(
            output_dir, lon, lat, noseed, seed, enhancement,
            float(lon_full[center_j, center_i]),
            float(lat_full[center_j, center_i]),
            case_name,
        )
    else:
        print("[8/8] 그림 저장 생략")

    print("-" * 90)
    print(f"대관령 최근접 WRF: j={center_j}, i={center_i}, "
          f"lat={lat_full[center_j, center_i]:.5f}, lon={lon_full[center_j, center_i]:.5f}")
    print(f"100×100 SEED 평균강수량       : {seed_mean:.4f} mm")
    print(f"100×100 NOSEED 평균강수량     : {noseed_mean:.4f} mm")
    print(f"100×100 영역 평균 증우량      : {enhancement_mean:+.4f} mm")
    print(f"100×100 영역 평균 증우율      : {enhancement_rate_100x100:+.4f} %")
    print(f"대관령 최근접 격자 증우율      : {station_enhancement_rate:+.4f} %")
    print(f"증우/감우 영역                : {positive_area_km2:.3f} / {negative_area_km2:.3f} km²")
    print(f"순 증우 수량                  : {net_enhanced_volume_m3:+,.3f} m³(≈ton)")
    print(f"대관령 ASOS 효과시간 누적강수 : {asos_precip:.4f} mm")
    print(f"ASOS 기반 추정 증우량         : {asos_estimated_enhancement:+.4f} mm")
    print(f"ASOS 기반 추정 증우량(최근접 격자 증우율): {asos_estimated_enhancement_station_rate:+.4f} mm")
    print(f"저장 위치                     : {output_dir.resolve()}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="대관령 ASOS 중심 100×100 WRF SEED/NOSEED 강수 및 증우량을 산출합니다."
    )
    parser.add_argument(
        "--asos-dir",
        type=Path,
        default=None,
        help="ASOS 파일 디렉터리. 지정하지 않으면 스크립트 상단 ASOS_DIR을 사용합니다.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="결과 저장 디렉터리. 지정하지 않으면 스크립트 상단 OUTPUT_ROOT를 사용합니다.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="PNG 그림 저장을 생략합니다.",
    )
    return parser.parse_args()


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    global ASOS_DIR, OUTPUT_ROOT, SAVE_FIGURES

    if args.asos_dir is not None:
        ASOS_DIR = args.asos_dir
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root
    if args.no_figures:
        SAVE_FIGURES = False


def main() -> int:
    args = parse_args()
    apply_runtime_overrides(args)
    ensure_directory(OUTPUT_ROOT)

    print("ASOS 디렉터리 전체 자료 읽기...")
    try:
        asos_df = read_asos_directory(ASOS_DIR, ASOS_STATION_ID)
    except Exception as exc:
        print(f"[ERROR] ASOS 읽기 실패: {exc}", file=sys.stderr)
        return 1

    all_summaries = []
    failed_cases = []

    for case in CASES:
        try:
            summary = process_case(case, asos_df)
            all_summaries.append(summary)
        except Exception as exc:
            case_name = case.get("case_name", "UNKNOWN")
            failed_cases.append({"case_name": case_name, "error": str(exc)})
            print(f"\n[ERROR] {case_name} 처리 실패:\n{exc}\n", file=sys.stderr)

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            OUTPUT_ROOT / "all_cases_summary.csv", index=False, encoding="utf-8-sig"
        )

    if failed_cases:
        pd.DataFrame(failed_cases).to_csv(
            OUTPUT_ROOT / "failed_cases.csv", index=False, encoding="utf-8-sig"
        )

    print("\n" + "=" * 90)
    print(f"성공 사례 수: {len(all_summaries)}")
    print(f"실패 사례 수: {len(failed_cases)}")
    print(f"전체 결과 디렉터리: {OUTPUT_ROOT.resolve()}")
    print("=" * 90)

    return 0 if not failed_cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
