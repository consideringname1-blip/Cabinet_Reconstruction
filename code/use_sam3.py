from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch

model = build_sam3_image_model().eval().cuda()
processor = Sam3Processor(model)

print("模型加载成功，检查 ~/.cache/huggingface/hub 是否有文件")
