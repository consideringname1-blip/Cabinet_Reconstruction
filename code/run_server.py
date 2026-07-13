import subprocess
import signal
import shlex
import json
import socket
import sys
from urllib.request import urlopen
from config import IS_RUN_FLASK_SERVER
from path_config import (
    CODE_ROOT,
    PROJECT_ROOT,
    HOLOLENS2_PY,
    SERVER_API_RUN,
    SERVER_PY,
    HOLOLENS2_DOWNLOAD_RUN,
    HOLOLENS2_DOWNLOAD_DIR,
)
from pathlib import Path

PROJECT_SETUP_ENV = PROJECT_ROOT / "setup_env.sh"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7355


def _probe_existing_server():
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1.0):
            pass
    except OSError:
        return "available"

    try:
        with urlopen(f"http://{SERVER_HOST}:{SERVER_PORT}/", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") == "running" and "/generate" in payload.get("endpoints", []):
            return "running"
    except Exception:
        pass
    return "occupied"


def _source_project_env_cmd(cmd):
    if not PROJECT_SETUP_ENV.exists():
        return cmd
    quoted_cmd = " ".join(shlex.quote(str(part)) for part in cmd)
    return [
        "bash",
        "-lc",
        f"source {shlex.quote(str(PROJECT_SETUP_ENV))} && exec {quoted_cmd}",
    ]


def _run_child(cmd, cwd):
    process = subprocess.Popen(_source_project_env_cmd(cmd), cwd=cwd)

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
        server_state = _probe_existing_server()
        if server_state == "running":
            print(f">>> server_api.py 已在运行: http://0.0.0.0:{SERVER_PORT}")
            return 0
        if server_state == "occupied":
            print(f">>> 无法启动: 端口 {SERVER_PORT} 已被其他程序占用")
            return 1

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
