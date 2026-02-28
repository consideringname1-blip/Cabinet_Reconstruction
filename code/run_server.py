import subprocess
from pathlib import Path
from config import SERVER_PY

def main():
    # generateModel.py 的路径由相对目录推导，不写死
    base_dir = Path(__file__).resolve().parent     # /workspace/code
    generate_model = base_dir / "generateModel.py"

    cmd = [
        SERVER_PY,          # 从 config.py 读取 server 环境 python
        str(generate_model) # 运行 generateModel.py
    ]

    print(">>> 使用 server 环境启动 generateModel.py")
    print(">>> CMD:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
