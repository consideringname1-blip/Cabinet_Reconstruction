import sys
from pathlib import Path

from _bootstrap import CODE_ROOT

SAM3_ROOT = CODE_ROOT / "reconstruction" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

model = build_sam3_image_model().eval().cuda()
processor = Sam3Processor(model)

print("模型加载成功，检查 ~/.cache/huggingface/hub 是否有文件")
