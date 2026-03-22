# torch_cuda_version.py
import os
# 一定要放在最前面！！！比任何 import 都前面
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 現在才開始 import
import torch

def check_pytorch_cuda():
    print("\n" + "="*60)
    print(" PyTorch 與 CUDA 環境檢測")
    print("="*60)

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"PyTorch 編譯時的 CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 設備數量: {torch.cuda.device_count()}")
        print(f"當前設備: {torch.cuda.get_device_name(0)}")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    else:
        print("CUDA 不可用，請檢查驅動 / CUDA Toolkit / PyTorch 安裝版本是否匹配")

    print("="*60 + "\n")

if __name__ == "__main__":
    check_pytorch_cuda()