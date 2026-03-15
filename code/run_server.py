import subprocess
from config import (
    CODE_ROOT,
    FLASK_SERVER,
    IS_RUN_FLASK_SERVER,
    HOLOLENS2_PY,
    SERVER_PY,
    HOLOLENS2_DOWNLOAD_RUN,
    HOLOLENS2_DOWNLOAD_DIR,
)
from pathlib import Path

def main():
    if(IS_RUN_FLASK_SERVER):
        cmd = [
            SERVER_PY,          # 从 config.py 读取 server 环境 python
            str(FLASK_SERVER) # 运行 generateModel.py
        ]

        print(">>> 使用 server 环境启动 generateModel.py")
        print(">>> CMD:", " ".join(cmd))
        subprocess.run(cmd, cwd=CODE_ROOT)
    else:
        cmd = [
            HOLOLENS2_PY,          # 从 config.py 读取 hololens2 的 server 环境 python
            str(HOLOLENS2_DOWNLOAD_RUN) # 运行 download_calibration_all.py
        ]

        print(">>> 使用 hololens2 server 环境启动 download_calibration_all.py")
        print(">>> CMD:", " ".join(cmd))
        subprocess.run(cmd, cwd=HOLOLENS2_DOWNLOAD_DIR)
        
if __name__ == "__main__":
    main()
