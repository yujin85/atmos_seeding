; -*- mode: ncl -*-
; 대관령 ASOS 중심 100x100 WRF SEED/NOSEED 강수 및 증우량 산출용 NCL 버전
;
; 참고 NCL: https://github.com/yujin85/atmos_seeding/blob/main/draw_anal_SUNNY.txt
; - RAINNC 시간 차분 기반 SEED-CNTL 강수 차이도
; - KOREA_MAP 경계선 오버레이
; - 명시적 증우량 등치선 및 색상표
;
; 실행 예시:
;   env \
;     NOSEED_FILE=/path/wrfout_xxx_NOSEED.nc \
;     SEED_FILE=/path/wrfout_xxx_SEED.nc \
;     EFFECT_START_KST="2025-09-17 14:00:00" \
;     EFFECT_END_KST="2025-09-17 18:00:00" \
;     OUTROOT=./WRF_ASOS_PRECIP_ENHANCEMENT/case_20250916 \
;     KOREA_MAP=/path/korea_map.txt \
;     ncl calculate_wrf_asos_precip_enhancement_single_nc.ncl
;
; 필수 환경변수:
;   NOSEED_FILE, SEED_FILE
; 선택 환경변수:
;   EFFECT_START_KST, EFFECT_END_KST  : Times와 매칭할 KST 시각
;   START_INDEX, END_INDEX            : EFFECT_* 대신 직접 Time index 지정
;   OUTROOT, CASE_NAME, KOREA_MAP

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

undef("envstr")
function envstr(name:string, default_value:string)
local value
begin
  value = getenv(name)
  if (ismissing(value) .or. value .eq. "") then
    return(default_value)
  end if
  return(value)
end

undef("envint")
function envint(name:string, default_value:integer)
local value
begin
  value = getenv(name)
  if (ismissing(value) .or. value .eq. "") then
    return(default_value)
  end if
  return(toint(value))
end

undef("format_time_string")
function format_time_string(parts[*]:numeric)
local year, month, day, hour, minute, second
begin
  year = toint(parts(0))
  month = toint(parts(1))
  day = toint(parts(2))
  hour = toint(parts(3))
  minute = toint(parts(4))
  second = toint(parts(5))
  return(sprinti("%0.4i", year) + "-" + sprinti("%0.2i", month) + "-" + \
         sprinti("%0.2i", day) + " " + sprinti("%0.2i", hour) + ":" + \
         sprinti("%0.2i", minute) + ":" + sprinti("%0.2i", second))
end

undef("wrf_time_utc_to_kst_strings")
function wrf_time_utc_to_kst_strings(times[*]:string)
local nt, out, i, y, mo, d, h, mi, s, units, hours, parts
begin
  nt = dimsizes(times)
  out = new(nt, string)
  units = "hours since 1900-01-01 00:00:00"
  do i = 0, nt - 1
    y = toint(str_get_cols(times(i), 0, 3))
    mo = toint(str_get_cols(times(i), 5, 6))
    d = toint(str_get_cols(times(i), 8, 9))
    h = toint(str_get_cols(times(i), 11, 12))
    mi = toint(str_get_cols(times(i), 14, 15))
    s = toint(str_get_cols(times(i), 17, 18))
    hours = cd_inv_calendar(y, mo, d, h, mi, s, units, 0) + 9.0d
    parts = cd_calendar(hours, 0)
    out(i) = format_time_string(parts)
  end do
  return(out)
end

undef("find_time_index")
function find_time_index(kst_times[*]:string, target:string)
local idx
begin
  idx = ind(kst_times .eq. target)
  if (ismissing(idx) .or. dimsizes(idx) .lt. 1) then
    print("[FATAL] requested KST time not found: " + target)
    print("        available range: " + kst_times(0) + " ~ " + kst_times(dimsizes(kst_times)-1))
    exit
  end if
  return(toint(idx(0)))
end

undef("period_rain")
function period_rain(f:file, start_idx:integer, end_idx:integer)
local rainc, rainnc, rain, out, it, raw_inc, reset_mask, tiny_neg_mask
begin
  if (end_idx .le. start_idx) then
    print("[FATAL] END index must be larger than START index")
    exit
  end if

  rainc = f->RAINC(start_idx:end_idx,:,:)
  rainnc = f->RAINNC(start_idx:end_idx,:,:)
  rain = rainc + rainnc
  rain = where(rain .lt. 0.0, 0.0, rain)

  out = rain(0,:,:)
  out = 0.0
  do it = 1, end_idx - start_idx
    raw_inc = rain(it,:,:) - rain(it-1,:,:)
    reset_mask = raw_inc .lt. -1.0e-6
    tiny_neg_mask = (raw_inc .lt. 0.0) .and. (.not. reset_mask)
    raw_inc = where(reset_mask, rain(it,:,:), raw_inc)
    raw_inc = where(tiny_neg_mask, 0.0, raw_inc)
    out = out + raw_inc
  end do

  copy_VarCoords(rain(0,:,:), out)
  return(out)
end

undef("nearest_grid_ij")
function nearest_grid_ij(lat2d[*][*]:numeric, lon2d[*][*]:numeric, target_lat:numeric, target_lon:numeric)
local lon_weight, dist2, flat, ij, nx, out
begin
  lon_weight = cos(0.0174532925199433 * target_lat)
  dist2 = (lat2d - target_lat)^2 + ((lon2d - target_lon) * lon_weight)^2
  flat = ind_resolve(minind(ndtooned(dist2)), dimsizes(dist2))
  nx = dimsizes(lat2d)(1)
  out = new(2, integer)
  out(0) = toint(flat(0,0))
  out(1) = toint(flat(0,1))
  return(out)
end

undef("subdomain_bounds")
function subdomain_bounds(center_j:integer, center_i:integer, ny:integer, nx:integer, sub_ny:integer, sub_nx:integer)
local out, j0, i0
begin
  if (sub_ny .gt. ny .or. sub_nx .gt. nx) then
    print("[FATAL] requested subdomain is larger than WRF domain")
    exit
  end if
  j0 = center_j - sub_ny / 2
  i0 = center_i - sub_nx / 2
  j0 = max((/0, min((/j0, ny - sub_ny/))/))
  i0 = max((/0, min((/i0, nx - sub_nx/))/))
  out = new(4, integer)
  out = (/j0, j0 + sub_ny - 1, i0, i0 + sub_nx - 1/)
  return(out)
end

undef("add_korea_map_lines")
procedure add_korea_map_lines(wks:graphic, plot:graphic, korea_map_file:string)
local lines, delim, respg, dum, j, i, num_l, info, line_lon, line_lat
begin
  if (ismissing(korea_map_file) .or. korea_map_file .eq. "") then
    return
  end if
  if (.not. fileexists(korea_map_file)) then
    print("[WARN] KOREA_MAP file not found; skip map lines: " + korea_map_file)
    return
  end if

  lines = asciiread(korea_map_file, -1, "string")
  delim = " "
  respg = True
  respg@gsLineColor = "black"
  respg@gsLineThicknessF = 2.0
  dum = new(10000, graphic)
  j = 0
  i = 0
  do while (i .lt. dimsizes(lines))
    num_l = toint(str_get_field(lines(i), 1, delim))
    info = toint(str_get_field(lines(i), 2, delim))
    if (.not. ismissing(num_l) .and. num_l .gt. 1) then
      if (info .eq. 0 .or. info .eq. 2) then
        line_lon = tofloat(str_get_field(lines(i+1:i+num_l), 1, delim))
        line_lat = tofloat(str_get_field(lines(i+1:i+num_l), 2, delim))
        dum(j) = gsn_add_polyline(wks, plot, line_lon, line_lat, respg)
        j = j + 1
        delete(line_lon)
        delete(line_lat)
      end if
      i = i + num_l + 1
    else
      i = i + 1
    end if
  end do
end

undef("base_map_res")
function base_map_res(lat2d[*][*]:numeric, lon2d[*][*]:numeric)
local res
begin
  res = True
  res@gsnFrame = False
  res@gsnDraw = False
  res@mpOutlineOn = False
  res@mpOutlineBoundarySets = "NoBoundaries"
  res@mpGridAndLimbDrawOrder = "Predraw"
  res@mpGridSpacingF = 1.0
  res@pmTickMarkDisplayMode = "Never"
  res@mpFillOn = True
  res@mpLandFillColor = "white"
  res@mpOceanFillColor = "white"
  res@mpInlandWaterFillColor = "white"
  res@mpLimitMode = "Corners"
  res@mpLeftCornerLatF = min(lat2d)
  res@mpLeftCornerLonF = min(lon2d)
  res@mpRightCornerLatF = max(lat2d)
  res@mpRightCornerLonF = max(lon2d)
  res@tfDoNDCOverlay = True
  res@cnFillOn = True
  res@cnLinesOn = False
  res@cnLineLabelsOn = False
  res@cnInfoLabelOn = False
  res@cnLevelSelectionMode = "ExplicitLevels"
  res@lbLabelBarOn = True
  res@lbAutoManage = False
  res@lbLabelFont = "helvetica-bold"
  res@lbLabelJust = "CenterRight"
  res@lbLabelFontHeightF = 0.010
  res@gsnLeftString = ""
  res@gsnRightString = "mm"
  return(res)
end

undef("draw_precip_map")
procedure draw_precip_map(values[*][*]:numeric, lat2d[*][*]:numeric, lon2d[*][*]:numeric, \
                          outname:string, title:string, levels[*]:numeric, colors[*], \
                          target_lat:numeric, target_lon:numeric, center_lat:numeric, center_lon:numeric, \
                          korea_map_file:string)
local wks_type, wks, res, plot, marker_asos, marker_grid, draw_values, dummy_asos, dummy_grid
begin
  draw_values = values
  draw_values@lat2d = lat2d
  draw_values@lon2d = lon2d

  wks_type = "png"
  wks_type@wkHeight = 1500
  wks_type@wkWidth = 1500
  wks = gsn_open_wks(wks_type, outname)

  res = base_map_res(lat2d, lon2d)
  res@tiMainString = title
  res@cnLevels := levels
  res@cnFillColors := colors

  plot = gsn_csm_contour_map(wks, draw_values, res)
  add_korea_map_lines(wks, plot, korea_map_file)

  marker_asos = True
  marker_asos@gsMarkerIndex = 16
  marker_asos@gsMarkerSizeF = 0.012
  marker_asos@gsMarkerColor = "red"
  dummy_asos = gsn_add_polymarker(wks, plot, target_lon, target_lat, marker_asos)

  marker_grid = True
  marker_grid@gsMarkerIndex = 4
  marker_grid@gsMarkerSizeF = 0.010
  marker_grid@gsMarkerColor = "black"
  dummy_grid = gsn_add_polymarker(wks, plot, center_lon, center_lat, marker_grid)

  draw(plot)
  frame(wks)
  delete(plot)
  delete(wks)
end

begin
  ; ---------------------------------------------------------------------------
  ; User/environment configuration
  ; ---------------------------------------------------------------------------
  noseed_file = envstr("NOSEED_FILE", "")
  seed_file = envstr("SEED_FILE", "")
  if (noseed_file .eq. "" .or. seed_file .eq. "") then
    print("[FATAL] NOSEED_FILE and SEED_FILE environment variables are required")
    exit
  end if

  outroot = envstr("OUTROOT", "./WRF_ASOS_PRECIP_ENHANCEMENT_NCL")
  case_name = envstr("CASE_NAME", "case_ncl")
  system("mkdir -p " + outroot)

  effect_start_kst = envstr("EFFECT_START_KST", "")
  effect_end_kst = envstr("EFFECT_END_KST", "")
  start_index = envint("START_INDEX", -1)
  end_index = envint("END_INDEX", -1)

  target_lat = stringtofloat(envstr("ASOS_LAT", "37.67713"))
  target_lon = stringtofloat(envstr("ASOS_LON", "128.71834"))
  sub_ny = envint("SUBDOMAIN_NY", 100)
  sub_nx = envint("SUBDOMAIN_NX", 100)
  korea_map_file = envstr("KOREA_MAP", "")

  f_noseed = addfile(noseed_file, "r")
  f_seed = addfile(seed_file, "r")

  times_utc = chartostring(f_seed->Times)
  times_kst = wrf_time_utc_to_kst_strings(times_utc)
  if (start_index .lt. 0 .or. end_index .lt. 0) then
    if (effect_start_kst .eq. "" .or. effect_end_kst .eq. "") then
      print("[FATAL] set either START_INDEX/END_INDEX or EFFECT_START_KST/EFFECT_END_KST")
      exit
    end if
    start_index = find_time_index(times_kst, effect_start_kst)
    end_index = find_time_index(times_kst, effect_end_kst)
  end if

  print("CASE                 : " + case_name)
  print("NOSEED_FILE          : " + noseed_file)
  print("SEED_FILE            : " + seed_file)
  print("Effect index         : " + start_index + " ~ " + end_index)
  print("Effect KST           : " + times_kst(start_index) + " ~ " + times_kst(end_index))

  lat_full = f_noseed->XLAT(0,:,:)
  lon_full = f_noseed->XLONG(0,:,:)
  ij = nearest_grid_ij(lat_full, lon_full, target_lat, target_lon)
  bounds = subdomain_bounds(ij(0), ij(1), dimsizes(lat_full)(0), dimsizes(lat_full)(1), sub_ny, sub_nx)

  noseed_full = period_rain(f_noseed, start_index, end_index)
  seed_full = period_rain(f_seed, start_index, end_index)
  enhancement_full = seed_full - noseed_full

  lat = lat_full(bounds(0):bounds(1), bounds(2):bounds(3))
  lon = lon_full(bounds(0):bounds(1), bounds(2):bounds(3))
  noseed = noseed_full(bounds(0):bounds(1), bounds(2):bounds(3))
  seed = seed_full(bounds(0):bounds(1), bounds(2):bounds(3))
  enhancement = enhancement_full(bounds(0):bounds(1), bounds(2):bounds(3))

  seed_mean = avg(seed)
  noseed_mean = avg(noseed)
  enhancement_mean = avg(enhancement)
  if (seed_mean .gt. 1.0e-12) then
    enhancement_rate = enhancement_mean / seed_mean * 100.0
  else
    enhancement_rate = default_fillvalue("float")
  end if

  ; CSV summary
  summary_file = outroot + "/summary_ncl.csv"
  header = "case_name,noseed_file,seed_file,start_index,end_index,effect_start_kst,effect_end_kst,nearest_wrf_j,nearest_wrf_i,nearest_wrf_lat,nearest_wrf_lon,seed_mean_100x100_mm,noseed_mean_100x100_mm,mean_enhancement_100x100_mm,enhancement_rate_100x100_percent"
  row = case_name + "," + noseed_file + "," + seed_file + "," + start_index + "," + end_index + "," + \
        times_kst(start_index) + "," + times_kst(end_index) + "," + ij(0) + "," + ij(1) + "," + \
        lat_full(ij(0),ij(1)) + "," + lon_full(ij(0),ij(1)) + "," + seed_mean + "," + \
        noseed_mean + "," + enhancement_mean + "," + enhancement_rate
  write_table(summary_file, "w", (/header/), "%s")
  write_table(summary_file, "a", (/row/), "%s")

  ; NCL 원본의 누적 증우량 차이도 levels/colors를 반영
  precip_levels = (/0,5,10,15,20,25,30,35,40,45,50,60,70,80,100/)
  precip_colors = span_color_rgba("MPL_gist_rainbow", dimsizes(precip_levels)+1)
  enh_levels = (/-3,-2,-1,-0.1,-0.01,0.01,0.1,1,2,3/)
  enh_colors = (/"mediumblue","mediumblue","dodgerblue","cadetblue1","lightcyan", \
                 "white","sandybrown","coral1","brown3","brown","brown"/)

  draw_precip_map(noseed, lat, lon, outroot + "/noseed_precip_ncl", \
                  case_name + " NOSEED accumulated precipitation", \
                  precip_levels, precip_colors, target_lat, target_lon, \
                  lat_full(ij(0),ij(1)), lon_full(ij(0),ij(1)), korea_map_file)

  draw_precip_map(seed, lat, lon, outroot + "/seed_precip_ncl", \
                  case_name + " SEED accumulated precipitation", \
                  precip_levels, precip_colors, target_lat, target_lon, \
                  lat_full(ij(0),ij(1)), lon_full(ij(0),ij(1)), korea_map_file)

  draw_precip_map(enhancement, lat, lon, outroot + "/enhancement_ncl", \
                  case_name + " Difference of Precipitation(SEED-NOSEED)[mm]", \
                  enh_levels, enh_colors, target_lat, target_lon, \
                  lat_full(ij(0),ij(1)), lon_full(ij(0),ij(1)), korea_map_file)

  print("100x100 SEED mean       : " + seed_mean + " mm")
  print("100x100 NOSEED mean     : " + noseed_mean + " mm")
  print("100x100 enhancement     : " + enhancement_mean + " mm")
  print("100x100 enhancement rate: " + enhancement_rate + " %")
  print("Output directory        : " + outroot)
end
