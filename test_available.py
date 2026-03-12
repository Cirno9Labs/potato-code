import os
# 关键：必须在 import torch 之前设置，解决 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torchvision import models

def test_env():
    print(f"--- 环境检查 ---")
    # 1. 检查 GPU (RTX 3060)
    gpu_ok = torch.cuda.is_available()
    print(f"GPU 是否可用: {gpu_ok}")
    if gpu_ok:
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")

    # 2. 检查 MobileNetV3 加载
    try:
        # 验证 torchvision 是否真的能找到 mobilenet_v3_large
        test_model = models.mobilenet_v3_large(weights=None)
        print("✅ MobileNetV3 模型结构加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

if __name__ == "__main__":
    test_env()