import subprocess
import signal
import sys
from config import (
    CODE_ROOT,
    IS_RUN_FLASK_SERVER,
    HOLOLENS2_PY,
    SERVER_API_RUN,
    SERVER_PY,
    HOLOLENS2_DOWNLOAD_RUN,
    HOLOLENS2_DOWNLOAD_DIR,
)
from pathlib import Path

def _run_child(cmd, cwd):
    process = subprocess.Popen(cmd, cwd=cwd)

    def _stop_child(signum=None, frame=None):
        if process.poll() is None:
            try:
                process.send_signal(signum or signal.SIGTERM)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if signum is not None:
            raise SystemExit(0)

    old_sigint = signal.signal(signal.SIGINT, _stop_child)
    old_sigterm = signal.signal(signal.SIGTERM, _stop_child)
    try:
        while True:
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        _stop_child()


def main():
    if(IS_RUN_FLASK_SERVER):
        cmd = [
            SERVER_PY,          # 从 config.py 读取 server 环境 python
            str(SERVER_API_RUN) # 运行 server_api.py
        ]

        print(">>> 使用 server 环境启动 server_api.py")
        print(">>> CMD:", " ".join(cmd))
        return _run_child(cmd, CODE_ROOT)
    else:
        cmd = [
            HOLOLENS2_PY,          # 从 config.py 读取 hololens2 的 server 环境 python
            str(HOLOLENS2_DOWNLOAD_RUN) # 运行 download_calibration_all.py
        ]

        print(">>> 使用 hololens2 server 环境启动 download_calibration_all.py")
        print(">>> CMD:", " ".join(cmd))
        return _run_child(cmd, HOLOLENS2_DOWNLOAD_DIR)
        
if __name__ == "__main__":
    raise SystemExit(main())
