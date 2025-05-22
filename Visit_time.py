import pandas as pd
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84
from numpy import dot, arccos
from numpy.linalg import norm
from datetime import timezone

from collections import defaultdict

# ----------- 参数设置 -----------
FOV_HALF_ANGLE_DEG = 5.0
FOV_HALF_ANGLE_RAD = FOV_HALF_ANGLE_DEG * 3.14159265 / 180
ORBIT_RESET_THRESHOLD_DEG = 3.0

# ----------- 文件路径 -----------
tle_path = '/Users/jianglinqiao/Desktop/Fire_location/satellites_10.tle'
coord_csv = '/Users/jianglinqiao/Desktop/Fire_location/Combine.csv'
output_path = '/Users/jianglinqiao/Desktop/Fire_location/satellite_pass_intervals_with_duration.csv'

# ----------- 时间设置（每分钟采样） -----------
ts = load.timescale()
start_dt = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
end_dt   = datetime(2025, 5, 7, 23, 59, tzinfo=timezone.utc)
minutes = int((end_dt - start_dt).total_seconds() // 60)

datetime_list = [start_dt + timedelta(minutes=i) for i in range(minutes)]
times = ts.utc([dt.year for dt in datetime_list],
               [dt.month for dt in datetime_list],
               [dt.day for dt in datetime_list],
               [dt.hour for dt in datetime_list],
               [dt.minute for dt in datetime_list],
               [dt.second for dt in datetime_list])

# ----------- 加载坐标数据 -----------
df = pd.read_csv(coord_csv)
locations = df[['lat', 'lon']].values.tolist()
fire_values = df['fire'].tolist()
user_requests = df['user_request'].tolist()

# ----------- 读取 TLE 数据 -----------
with open(tle_path, 'r') as f:
    lines = f.readlines()
tle_entries = [(lines[i].strip(), lines[i+1].strip(), lines[i+2].strip())
               for i in range(0, len(lines), 3)]

# ----------- 主循环：按卫星处理 -----------
print("📡 开始处理所有卫星...")
results = []

for idx, (name, line1, line2) in enumerate(tle_entries):
    sat = EarthSatellite(line1, line2, name, ts)
    print(f" - {idx+1:02d}. {name}")

    subpoints = sat.at(times).subpoint()
    sat_lons = subpoints.longitude.degrees

    # 估算 orbit_index（基于经度循环）
    orbit_index = 0
    orbit_flags = [0] * len(times)
    start_lon = sat_lons[0]
    for i in range(1, len(sat_lons)):
        delta = abs(sat_lons[i] - start_lon)
        if delta > 360 - ORBIT_RESET_THRESHOLD_DEG or delta < ORBIT_RESET_THRESHOLD_DEG:
            orbit_index += 1
            start_lon = sat_lons[i]
        orbit_flags[i] = orbit_index

    sat_positions = sat.at(times).position.km.T
    nadir = -sat_positions

    # 遍历每个地面点
    for i, (lat, lon) in enumerate(locations):
        target = wgs84.latlon(lat, lon).at(times)
        target_positions = target.position.km.T
        vec_to_target = target_positions - sat_positions

        cos_angle = [dot(a, b) / (norm(a) * norm(b)) for a, b in zip(nadir, vec_to_target)]
        angles_rad = arccos(cos_angle)
        visible = angles_rad < FOV_HALF_ANGLE_RAD

        # 检测所有 "可视区间"
        in_view = False
        start_time = None
        for j in range(len(visible)):
            if visible[j] and not in_view:
                in_view = True
                start_time = times[j].utc_datetime()
                start_orbit = orbit_flags[j]
            elif not visible[j] and in_view:
                end_time = times[j].utc_datetime()
                duration_min = round((end_time - start_time).total_seconds() / 60, 2)
                results.append({
                    "satellite": sat.name,
                    "orbit_index": start_orbit,
                    "location_index": i,
                    "lat": lat,
                    "lon": lon,
                    "fire": fire_values[i],
                    "user_request": user_requests[i],
                    "start_utc": start_time.isoformat() + "Z",
                    "end_utc": end_time.isoformat() + "Z",
                    "duration_min": duration_min
                })
                in_view = False

        if in_view:
            end_time = times[-1].utc_datetime()
            duration_min = round((end_time - start_time).total_seconds() / 60, 2)
            results.append({
                "satellite": sat.name,
                "orbit_index": start_orbit,
                "location_index": i,
                "lat": lat,
                "lon": lon,
                "fire": fire_values[i],
                "user_request": user_requests[i],
                "start_utc": start_time.isoformat() + "Z",
                "end_utc": end_time.isoformat() + "Z",
                "duration_min": duration_min
            })

# ----------- 保存输出 -----------
print(f"\n📊 统计完成：共记录 {len(results)} 次覆盖事件")
result_df = pd.DataFrame(results)
result_df.to_csv(output_path, index=False)
print(f"✅ 带时长的覆盖区间已保存至：{output_path}")
