import requests
import json
from pathlib import Path
from config import DATA_ROOT  # 使用你的 config.py 里定义的路径

# 服务器的接口地址
url = "http://10.40.1.122:7355/generate"

# 示例的 pose_depth 数据
pose_depth = {
    "depth": 0.8,
    "pos": {"x": 0.1, "y": 1.2, "z": -0.3},
    "rot": {"x": 0, "y": 0, "z": 0, "w": 1},
}

data = {
    "center_depth": "0.8",
    "pose_depth": json.dumps(pose_depth),
}

# 自动构造图片路径： /workspace/data/upload/test.png
# 不写死路径，更安全也更符合你现在的项目结构
image_path = DATA_ROOT / "upload" / "test.png"

# 打开文件
files = {
    "image": open(image_path, "rb"),
}

print(">>> 请求 URL:", url)
print(">>> 上传图片:", image_path)

# 发送 POST 请求
response = requests.post(url, data=data, files=files)

# 打印返回值
print(">>> 服务器返回：")
print(response.text)
