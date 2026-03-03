import os
import time
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import HOLOLENS2_HOST

host = HOLOLENS2_HOST

sockopt = None

out_dir = os.path.join("./hl2ss_calib", host)

os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# 1) AHAT depth
# -----------------------------
cal_ahat = hl2ss_3dcv.get_calibration_rm(
    out_dir,
    host,
    hl2ss.StreamPort.RM_DEPTH_AHAT,
    sockopt
)
print("RM_DEPTH_AHAT calibration OK")

# -----------------------------
# 2) Long Throw depth（可选）
# -----------------------------
cal_lt = hl2ss_3dcv.get_calibration_rm(
    out_dir,
    host,
    hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    sockopt
)
print("RM_DEPTH_LONGTHROW calibration OK")

# -----------------------------
# 3) VLC cameras（SLAM）
# -----------------------------
for port in [
    hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    hl2ss.StreamPort.RM_VLC_LEFTLEFT,
    hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    hl2ss.StreamPort.RM_VLC_RIGHTRIGHT,
]:
    _ = hl2ss_3dcv.get_calibration_rm(out_dir, host, port, sockopt)
    print(f"RM_VLC calibration OK: {hl2ss.get_port_name(port)}")

# -----------------------------
# 4) PV camera（重点）
# -----------------------------
pv_port = hl2ss.StreamPort.PERSONAL_VIDEO

# 你希望的 PV 参数
focus = 0
width = 1920
height = 1080
framerate = 30

# 某些版本需要显式启动 PV 子系统；有就调用，没有就跳过
if hasattr(hl2ss_lnm, "start_subsystem_pv"):
    try:
        hl2ss_lnm.start_subsystem_pv(host, pv_port, sockopt)
        # 你的版本没有 wait_for_pv_subsystem，所以简单 sleep
        time.sleep(1.0)
        print("PV subsystem started")
    except Exception as e:
        # 不要因为 start 失败就直接退出；继续尝试下载以便观察真实错误
        print(f"PV subsystem start failed (will still try calibration): {e}")

try:
    cal_pv = hl2ss_3dcv.get_calibration_pv(
        out_dir,
        host,
        pv_port,
        sockopt,
        focus,
        width,
        height,
        framerate
    )
    print("PV calibration OK")
except Exception as e:
    print(f"PV calibration FAILED: {e}")
    raise
finally:
    # 用完关掉（释放占用/锁）；有就调用，没有就跳过
    if hasattr(hl2ss_lnm, "stop_subsystem_pv"):
        try:
            hl2ss_lnm.stop_subsystem_pv(host, pv_port, sockopt)
            time.sleep(0.2)
            print("PV subsystem stopped")
        except Exception as e:
            print(f"PV subsystem stop failed: {e}")

print("All calibration downloaded to:", os.path.abspath(out_dir))
