#!/usr/bin/env bash
# NCL 버전 WRF-ASOS SEED/NOSEED 증우량 분석 실행 예시
#
# 사용 방법:
#   1) 이 파일을 복사합니다.
#        cp run_wrf_asos_precip_enhancement_ncl_example.sh run_case_20250916.sh
#   2) 아래 경로와 시간 값을 실제 사례에 맞게 수정합니다.
#   3) 실행합니다.
#        bash run_case_20250916.sh
#
# 주의:
#   - NOSEED_FILE, SEED_FILE은 실제 wrfout NetCDF 파일 전체 경로입니다.
#   - EFFECT_START_KST/EFFECT_END_KST는 WRF Times를 UTC+9(KST)로 바꾼 시각과
#     정확히 일치해야 합니다. 시간이 맞지 않으면 START_INDEX/END_INDEX를 대신 쓰세요.
#   - KOREA_MAP은 선택값입니다. 파일이 없으면 줄을 주석 처리하거나 빈 값으로 두세요.

set -euo pipefail

export NOSEED_FILE="/h3/home/nimr/yjinkim/fcst/MODL/RAMA_ANAL_GR/DAOU/2025091621/wrfout_example_NOSEED.nc"
export SEED_FILE="/h3/home/nimr/yjinkim/fcst/MODL/RAMA_ANAL_GR/DAOU/2025091621/wrfout_example_SEED.nc"

export CASE_NAME="case_20250916"
export OUTROOT="./WRF_ASOS_PRECIP_ENHANCEMENT_NCL/${CASE_NAME}"

# 방법 1: 효과시간을 KST 문자열로 지정합니다.
export EFFECT_START_KST="2025-09-17 14:00:00"
export EFFECT_END_KST="2025-09-17 18:00:00"

# 방법 2: 위 KST 문자열 매칭이 안 될 때 Time index를 직접 지정합니다.
# START_INDEX/END_INDEX를 쓰려면 아래 두 줄의 주석을 풀고 실제 index로 바꾸세요.
# export START_INDEX=14
# export END_INDEX=18

# 대관령 ASOS 좌표 및 분석 격자 크기입니다. 다른 지점이면 수정하세요.
export ASOS_LAT="37.67713"
export ASOS_LON="128.71834"
export SUBDOMAIN_NY=100
export SUBDOMAIN_NX=100

# 선택: draw_anal_SUNNY.ncl에서 사용하던 KOREA_MAP 경계선 파일 경로입니다.
# 파일이 없으면 빈 값으로 두면 됩니다.
export KOREA_MAP="${KOREA_MAP:-}"

ncl calculate_wrf_asos_precip_enhancement_single_nc.ncl
