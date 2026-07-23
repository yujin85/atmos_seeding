#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대관령 ASOS 중심 100×100 WRF 격자의 SEED/NOSEED 강수 및 증우량 산출

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
    아래 ASOS_DIR, OUTPUT_ROOT 및 CASES 설정을 실제 환경에 맞게 수정한다.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# 사용자 설정
# =============================================================================

# 대관령 ASOS
ASOS_STATION_ID = 100
ASOS_STATION_NAME = "Daegwallyeong"
ASOS_LAT = 37.67713
ASOS_LON = 128.71834

# ASOS 시간자료 디렉터리
ASOS_DIR = Path(
    "/h3/home/nimr/yjinkim/_data/WRF_initial_data/OBS_data/DAIN/SFC_2025090100"
)
ASOS_GLOB = "KMA_ASOS_*.dat"

# 결과 최상위 디렉터리
OUTPUT_ROOT = Path("./WRF_ASOS_PRECIP_ENHANCEMENT")

# 추출 격자 수
SUBDOMAIN_NY = 100
SUBDOMAIN_NX = 100

# 동일 디렉터리 안에서 NOSEED/SEED 파일을 파일명으로 구분한다.
# 사례 설정은 wrf_dir(동일 디렉터리) 또는 noseed_dir/seed_dir(분리 디렉터리)을 지원한다.
NOSEED_FILE_PATTERN = "wrfout_*_NOSEED.nc"
SEED_FILE_PATTERN = "wrfout_*_SEED.nc"
WRF_RECURSIVE = False

# 쉘 파일에서 사례를 자동 생성할 때 사용하는 선택 설정
# True로 바꾸면 아래 CASES 목록 대신 SHELL_DIR/SHELL_GLOB에서 사례를 읽는다.
USE_SHELL_CASE_DISCOVERY = False
SHELL_DIR = Path("./case_shells")
SHELL_GLOB = "*.sh"
WRF_ROOT = Path(".")
NOSEED_RELATIVE_DIR = ""
SEED_RELATIVE_DIR = ""

# 효과시간 경계 시각 허용오차
WRF_TIME_TOLERANCE_MINUTES = 0
ASOS_TIME_TOLERANCE_MINUTES = 0

# 증우/감우 판정 임계값
ENHANCEMENT_THRESHOLD_MM = 0.01
MIN_SEED_MEAN_FOR_RATE_MM = 1.0e-12
SAVE_FIGURES = True

# -----------------------------------------------------------------------------
# NCL 유사 그림 설정
# -----------------------------------------------------------------------------
# SEED/NOSEED에는 반드시 같은 등치선 구간을 적용해 직접 비교한다.
PRECIP_LEVELS_MM = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100],
    dtype=float,
)
PRECIP_COLORMAP = "turbo"

# draw_anal_SUNNY.ncl의 누적 증우량 등치선 구간과 색상 구조를 반영한다.
ENHANCEMENT_LEVELS_MM = np.array(
    [-3, -2, -1, -0.1, -0.01, 0.01, 0.1, 1, 2, 3],
    dtype=float,
)
ENHANCEMENT_COLORS = [
    "mediumblue", "mediumblue", "dodgerblue", "cadetblue",
    "lightcyan", "white", "sandybrown", "coral",
    "brown", "darkred", "maroon",
]

# NCL에서 KOREA_MAP 환경변수로 읽던 행정경계 파일.
# 환경변수가 없으면 경계선을 생략하고 WRF 격자만 그린다.
KOREA_MAP_FILE = (
    Path(os.environ["KOREA_MAP"]) if os.environ.get("KOREA_MAP") else None
)
MAP_LINEWIDTH = 0.8
FIGURE_DPI = 250

# 사례별 설정
# - WRF 파일 내부 Times: UTC
# - seeding/effect 시간 입력값: KST
CASES = [
    {
        "case_name": "case_20250916",
        "wrf_dir": (
            "/h3/home/nimr/yjinkim/fcst/MODL/"
            "RAMA_ANAL_GR/DAOU/2025091621"
        ),
        "wrf_time_basis": "UTC",

        # 실제 UAV 시딩 수행시간(KST): 기록·출력용
        "seeding_start_kst": "2025-09-17 15:25:00",
        "seeding_end_kst": "2025-09-17 15:58:00",

        # 실제 강수/증우량 분석시간(KST)
        "effect_start_kst": "2025-09-17 14:00:00",
        "effect_end_kst": "2025-09-17 18:00:00",
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
        "wrf_time_basis",
        "seeding_start_kst",
        "seeding_end_kst",
        "effect_start_kst",
        "effect_end_kst",
    }
    missing = required.difference(case)
    if missing:
        raise KeyError(f"사례 설정 누락: {sorted(missing)}")

    has_shared_dir = bool(case.get("wrf_dir"))
    has_split_dirs = bool(case.get("noseed_dir")) and bool(case.get("seed_dir"))
    if not (has_shared_dir or has_split_dirs):
        raise KeyError("사례 설정에는 wrf_dir 또는 noseed_dir/seed_dir가 필요합니다.")

    seed_start = parse_datetime(case["seeding_start_kst"])
    seed_end = parse_datetime(case["seeding_end_kst"])
    effect_start = parse_datetime(case["effect_start_kst"])
    effect_end = parse_datetime(case["effect_end_kst"])

    if seed_end <= seed_start:
        raise ValueError(f"{case['case_name']}: 시딩 종료가 시작보다 늦어야 합니다.")
    if effect_end <= effect_start:
        raise ValueError(f"{case['case_name']}: 효과 종료가 시작보다 늦어야 합니다.")


def utc_to_kst(value: datetime) -> datetime:
    """UTC datetime을 KST(UTC+9)로 변환한다."""
    return value + timedelta(hours=9)


def read_shell_case_info(shell_file: Path) -> dict:
    """쉘 파일의 TIME, effect_start, effect_end를 읽고 UTC/KST 시각을 생성한다."""
    text = Path(shell_file).read_text(encoding="utf-8", errors="replace")
    patterns = {
        "time": r"^\s*(?:export\s+)?TIME\s*=\s*['\"]?(\d{10})['\"]?",
        "effect_start": r"^\s*(?:export\s+)?effect_start\s*=\s*['\"]?(\d{10})['\"]?",
        "effect_end": r"^\s*(?:export\s+)?effect_end\s*=\s*['\"]?(\d{10})['\"]?",
    }
    values = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches:
            raise ValueError(f"{shell_file.name}: {key} 값을 찾지 못했습니다.")
        values[key] = matches[-1]

    model_start_utc = datetime.strptime(values["time"], "%Y%m%d%H")
    effect_start_utc = datetime.strptime(values["effect_start"], "%Y%m%d%H")
    effect_end_utc = datetime.strptime(values["effect_end"], "%Y%m%d%H")
    if effect_end_utc <= effect_start_utc:
        raise ValueError(f"{shell_file.name}: effect_end가 effect_start보다 늦어야 합니다.")

    return {
        "time_dir": values["time"],
        "model_start_utc": model_start_utc,
        "model_start_kst": utc_to_kst(model_start_utc),
        "effect_start_utc": effect_start_utc,
        "effect_end_utc": effect_end_utc,
        "effect_start_kst": utc_to_kst(effect_start_utc),
        "effect_end_kst": utc_to_kst(effect_end_utc),
    }


def make_case_from_shell(shell_file: Path) -> dict:
    """쉘의 TIME을 디렉터리명으로 사용해 사례 설정을 자동 생성한다."""
    info = read_shell_case_info(shell_file)
    case_root = WRF_ROOT / info["time_dir"]
    noseed_dir = case_root / NOSEED_RELATIVE_DIR if NOSEED_RELATIVE_DIR else case_root
    seed_dir = case_root / SEED_RELATIVE_DIR if SEED_RELATIVE_DIR else case_root

    return {
        "case_name": shell_file.stem,
        "shell_file": str(shell_file),
        "time_dir": info["time_dir"],
        "model_start_utc": info["model_start_utc"].strftime("%Y-%m-%d %H:%M:%S"),
        "model_start_kst": info["model_start_kst"].strftime("%Y-%m-%d %H:%M:%S"),
        "noseed_dir": str(noseed_dir),
        "seed_dir": str(seed_dir),
        "wrf_time_basis": "UTC",
        # 시딩 수행시간은 이 분석에 필수 값이 아니므로 효과시간으로 대체 기록
        "seeding_start_kst": info["effect_start_kst"].strftime("%Y-%m-%d %H:%M:%S"),
        "seeding_end_kst": info["effect_end_kst"].strftime("%Y-%m-%d %H:%M:%S"),
        "effect_start_utc": info["effect_start_utc"].strftime("%Y-%m-%d %H:%M:%S"),
        "effect_end_utc": info["effect_end_utc"].strftime("%Y-%m-%d %H:%M:%S"),
        "effect_start_kst": info["effect_start_kst"].strftime("%Y-%m-%d %H:%M:%S"),
        "effect_end_kst": info["effect_end_kst"].strftime("%Y-%m-%d %H:%M:%S"),
    }


def discover_cases_from_shells() -> List[dict]:
    """SHELL_DIR의 모든 사례 쉘을 읽어 CASES 목록을 자동 생성한다."""
    if not SHELL_DIR.exists():
        raise FileNotFoundError(f"쉘 디렉터리가 없습니다: {SHELL_DIR}")
    shell_files = sorted(p for p in SHELL_DIR.glob(SHELL_GLOB) if p.is_file())
    if not shell_files:
        raise FileNotFoundError(f"쉘 파일을 찾지 못했습니다: {SHELL_DIR}/{SHELL_GLOB}")
    return [make_case_from_shell(path) for path in shell_files]


def get_configured_cases() -> List[dict]:
    """설정 방식에 따라 수동 CASES 또는 쉘 자동 생성 사례 목록을 반환한다."""
    if USE_SHELL_CASE_DISCOVERY:
        return discover_cases_from_shells()
    return list(CASES)


def describe_case_wrf_dirs(case: dict) -> str:
    """시작 로그에 표시할 WRF 디렉터리 설정을 문자열로 만든다."""
    if case.get("wrf_dir"):
        return str(case["wrf_dir"])
    return f"{case.get('noseed_dir', '')} / {case.get('seed_dir', '')}"


# =============================================================================
# WRF 시간 및 변수 읽기
# =============================================================================

def decode_wrf_times(ds: xr.Dataset) -> List[datetime]:
    """WRF Times 또는 XTIME을 실제 datetime 목록으로 읽는다."""
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


def find_single_wrf_file(directory: Path, pattern: str, label: str) -> Path:
    """동일 디렉터리에서 지정 패턴과 일치하는 WRF 파일 하나를 찾는다."""
    if not directory.exists():
        raise FileNotFoundError(f"WRF 디렉터리가 없습니다: {directory}")

    iterator = directory.rglob(pattern) if WRF_RECURSIVE else directory.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())

    if not files:
        raise FileNotFoundError(
            f"{label} WRF 파일을 찾지 못했습니다: {directory}/{pattern}"
        )
    if len(files) > 1:
        file_list = "\n".join(f"  - {path}" for path in files)
        raise RuntimeError(
            f"{label} WRF 파일이 여러 개 검색되었습니다. 패턴을 더 정확히 지정하십시오.\n"
            f"{file_list}"
        )
    return files[0]


def build_wrf_time_index(file_path: Path, time_basis: str) -> Dict[datetime, WRFTimeRecord]:
    """단일 WRF NetCDF 파일 내부의 모든 Time을 KST 기준으로 색인한다."""
    if not file_path.exists():
        raise FileNotFoundError(f"WRF 파일이 없습니다: {file_path}")

    try:
        with xr.open_dataset(file_path, decode_times=False, mask_and_scale=False) as ds:
            native_times = decode_wrf_times(ds)
    except Exception as exc:
        raise RuntimeError(f"WRF 시간 읽기 실패: {file_path}\n{exc}") from exc

    index: Dict[datetime, WRFTimeRecord] = {}
    for time_index, native_time in enumerate(native_times):
        kst_time = native_to_kst(native_time, time_basis)
        if kst_time in index:
            raise ValueError(
                f"{file_path}: 파일 내부에 중복 WRF 유효시각이 있습니다: {kst_time}"
            )
        index[kst_time] = WRFTimeRecord(
            valid_time_kst=kst_time,
            valid_time_native=native_time,
            file_path=file_path,
            time_index=time_index,
        )

    if not index:
        raise ValueError(f"{file_path}: WRF Time 자료가 없습니다.")
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
    """격자면적(m²)을 계산한다. MAPFAC_M이 있으면 투영 보정을 적용한다."""
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

        return np.full(shape, dx * dy, dtype=np.float64)


# =============================================================================
# ASOS 읽기
# =============================================================================

ASOS_COLUMNS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT",
    "PR", "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP",
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]


def parse_asos_data_line(line: str) -> Optional[dict]:
    """
    기상청 ASOS help=1 텍스트 한 줄을 파싱한다.

    WW와 CT가 문자열이고 공백 폭이 고정되어 있어 단순 split만으로는 항상 45열이
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
    if target in available_times:
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
                else:
                    # 같은 날짜인데 누적값이 감소하면 품질 문제로 RN 대체 시도
                    if pd.notna(curr["RN"]) and curr["RN"] >= 0.0:
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


def read_korea_map_segments(path: Optional[Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    NCL KOREA_MAP 형식의 경계선 파일을 읽는다.

    각 블록의 첫 줄은 "점개수 경계종류", 이후 점개수만큼 "경도 위도"가
    이어지는 형식을 가정한다. NCL 코드와 동일하게 경계종류 0 또는 2만 그린다.
    """
    if path is None or not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        if len(tokens) < 2:
            i += 1
            continue
        try:
            npoint = int(tokens[0])
            info = int(tokens[1])
        except ValueError:
            i += 1
            continue

        coords = []
        for row in lines[i + 1:i + 1 + npoint]:
            values = row.split()
            if len(values) < 2:
                continue
            try:
                coords.append((float(values[0]), float(values[1])))
            except ValueError:
                continue

        if info in {0, 2} and len(coords) >= 2:
            arr = np.asarray(coords, dtype=float)
            segments.append((arr[:, 0], arr[:, 1]))
        i += npoint + 1

    return segments


def add_map_lines(
    ax: plt.Axes,
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """분석영역 안에 포함되는 해안선·행정경계선을 추가한다."""
    for line_lon, line_lat in segments:
        ax.plot(
            line_lon, line_lat,
            color="black", linewidth=MAP_LINEWIDTH,
            solid_capstyle="round", zorder=5,
        )


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
    """
    NCL 그림과 유사하게 이산 등치선, 공통 강수 색상범위, 행정경계선 및
    대관령 표식을 적용하여 SEED/NOSEED/증우량 지도를 저장한다.
    """
    map_segments = read_korea_map_segments(KOREA_MAP_FILE)

    precip_cmap = plt.get_cmap(PRECIP_COLORMAP, len(PRECIP_LEVELS_MM) - 1)
    precip_norm = BoundaryNorm(PRECIP_LEVELS_MM, precip_cmap.N, clip=False)

    enhancement_cmap = ListedColormap(ENHANCEMENT_COLORS)
    enhancement_norm = BoundaryNorm(
        ENHANCEMENT_LEVELS_MM, enhancement_cmap.N, clip=False
    )

    plot_specs = [
        {
            "values": noseed,
            "title": "NOSEED accumulated precipitation",
            "filename": "noseed_precip_ncl_style.png",
            "levels": PRECIP_LEVELS_MM,
            "cmap": precip_cmap,
            "norm": precip_norm,
            "extend": "max",
        },
        {
            "values": seed,
            "title": "SEED accumulated precipitation",
            "filename": "seed_precip_ncl_style.png",
            "levels": PRECIP_LEVELS_MM,
            "cmap": precip_cmap,
            "norm": precip_norm,
            "extend": "max",
        },
        {
            "values": enhancement,
            "title": "Precipitation enhancement (SEED - NOSEED)",
            "filename": "enhancement_ncl_style.png",
            "levels": ENHANCEMENT_LEVELS_MM,
            "cmap": enhancement_cmap,
            "norm": enhancement_norm,
            "extend": "both",
        },
    ]

    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 8.5))

        contour = ax.contourf(
            lon, lat, spec["values"],
            levels=spec["levels"],
            cmap=spec["cmap"],
            norm=spec["norm"],
            extend=spec["extend"],
            antialiased=False,
            zorder=1,
        )

        add_map_lines(ax, map_segments)

        ax.scatter(
            ASOS_LON, ASOS_LAT, marker="*", s=145,
            facecolor="red", edgecolor="black", linewidth=0.7,
            label=f"{ASOS_STATION_NAME} ASOS ({ASOS_STATION_ID})", zorder=8,
        )
        ax.scatter(
            center_lon, center_lat, marker="o", s=38,
            facecolor="none", edgecolor="black", linewidth=1.0,
            label="Nearest WRF grid", zorder=8,
        )

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(1.0 / math.cos(math.radians((lat_min + lat_max) / 2.0)))
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{case_name}: {spec['title']}", fontsize=14, pad=10)
        ax.tick_params(direction="out", top=False, right=False)
        ax.grid(False)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

        cbar = fig.colorbar(
            contour, ax=ax, orientation="vertical",
            pad=0.025, fraction=0.048, ticks=spec["levels"],
        )
        cbar.set_label("mm")
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout()
        fig.savefig(
            output_dir / spec["filename"],
            dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white",
        )
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

    wrf_dir = Path(case["wrf_dir"]) if case.get("wrf_dir") else None
    noseed_dir = Path(case.get("noseed_dir") or wrf_dir)
    seed_dir = Path(case.get("seed_dir") or wrf_dir)
    time_basis = case["wrf_time_basis"]

    noseed_file = find_single_wrf_file(
        noseed_dir, NOSEED_FILE_PATTERN, "NOSEED"
    )
    seed_file = find_single_wrf_file(
        seed_dir, SEED_FILE_PATTERN, "SEED"
    )

    print("\n" + "=" * 90)
    print(f"CASE: {case_name}")
    print("=" * 90)
    print(f"Seeding KST : {seeding_start} ~ {seeding_end}")
    print(f"Effect  KST : {effect_start} ~ {effect_end}")

    if wrf_dir is not None:
        print(f"WRF directory: {wrf_dir}")
    else:
        print(f"NOSEED directory: {noseed_dir}")
        print(f"SEED directory  : {seed_dir}")
    print(f"NOSEED file : {noseed_file.name}")
    print(f"SEED file   : {seed_file.name}")

    print("[1/8] 단일 WRF 파일 내부의 전체 Time 색인 생성")
    noseed_index = build_wrf_time_index(noseed_file, time_basis)
    seed_index = build_wrf_time_index(seed_file, time_basis)

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
    lat_full, lon_full = read_lat_lon(noseed_records[0])
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

    print("[5/8] 같은 효과시간의 대관령 ASOS 누적강수량 계산")
    asos_precip, asos_period, asos_method = calculate_asos_period_precipitation(
        asos_df, actual_effect_start, actual_effect_end, ASOS_TIME_TOLERANCE_MINUTES
    )

    if np.isfinite(enhancement_rate_100x100):
        asos_estimated_enhancement = asos_precip * enhancement_rate_100x100 / 100.0
    else:
        asos_estimated_enhancement = float("nan")

    print("[6/8] CSV 저장")
    summary = {
        "case_name": case_name,
        "shell_file": case.get("shell_file", ""),
        "time_dir": case.get("time_dir", ""),
        "model_start_utc": case.get("model_start_utc", ""),
        "model_start_kst": case.get("model_start_kst", ""),
        "wrf_dir": str(wrf_dir) if wrf_dir is not None else "",
        "noseed_dir": str(noseed_dir),
        "seed_dir": str(seed_dir),
        "noseed_file": str(noseed_file),
        "seed_file": str(seed_file),
        "effect_start_utc": case.get("effect_start_utc", ""),
        "effect_end_utc": case.get("effect_end_utc", ""),
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
        "analysis_area_km2": analysis_area_km2,
        "seed_mean_100x100_mm": seed_mean,
        "noseed_mean_100x100_mm": noseed_mean,
        "mean_enhancement_100x100_mm": enhancement_mean,
        "enhancement_rate_100x100_percent": enhancement_rate_100x100,
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
        key: (value.isoformat(sep=" ") if isinstance(value, datetime) else value)
        for key, value in summary.items()
        if value is not None and not isinstance(value, (np.generic,))
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
    print(f"증우/감우 영역                : {positive_area_km2:.3f} / {negative_area_km2:.3f} km²")
    print(f"순 증우 수량                  : {net_enhanced_volume_m3:+,.3f} m³(≈ton)")
    print(f"대관령 ASOS 효과시간 누적강수 : {asos_precip:.4f} mm")
    print(f"ASOS 기반 추정 증우량         : {asos_estimated_enhancement:+.4f} mm")
    print(f"저장 위치                     : {output_dir.resolve()}")

    return summary


def main() -> int:
    ensure_directory(OUTPUT_ROOT)

    try:
        cases = get_configured_cases()
    except Exception as exc:
        print(f"[ERROR] 사례 설정 생성 실패: {exc}", file=sys.stderr)
        return 1

    case_source = "쉘 자동 생성" if USE_SHELL_CASE_DISCOVERY else "수동 CASES"
    print(f"사례 설정 방식: {case_source}")
    print(f"설정된 사례 수: {len(cases)}")
    for case in cases:
        print(
            f"  {case['case_name']}: "
            f"WRF={describe_case_wrf_dirs(case)}, "
            f"effect KST={case['effect_start_kst']}~{case['effect_end_kst']}"
        )

    print("ASOS 디렉터리 전체 자료 읽기...")
    try:
        asos_df = read_asos_directory(ASOS_DIR, ASOS_STATION_ID)
    except Exception as exc:
        print(f"[ERROR] ASOS 읽기 실패: {exc}", file=sys.stderr)
        return 1

    all_summaries = []
    failed_cases = []

    for case in cases:
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
