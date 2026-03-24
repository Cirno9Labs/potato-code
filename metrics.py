import os
# 解决 Windows libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import math
import torch
import torch.nn as nn
import pandas as pd
from thop import profile
from torchvision import models
from torchvision.models.mobilenetv2 import InvertedResidual


# -----------------------------
# ECA模块（与你训练代码一致）
# -----------------------------
class ECAModule(nn.Module):

    def __init__(self, channels, gamma=2, b=1):

        super().__init__()

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k,padding=k//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1,-2)

        y = self.conv(y)

        y = y.transpose(-1,-2).unsqueeze(-1)

        return x * self.sigmoid(y).expand_as(x)


# -----------------------------
# MobileNetV2-ECA
# -----------------------------
class ImprovedMobileNetV2(nn.Module):

    def __init__(self,num_classes=3):

        super().__init__()

        base = models.mobilenet_v2(weights=None)

        self.features = base.features

        for module in self.features:

            if isinstance(module, InvertedResidual):

                out_channels = module.out_channels

                module.conv.add_module(
                    "eca",
                    ECAModule(out_channels)
                )

        last_channel = base.classifier[1].in_features

        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel,num_classes)

    def forward(self,x):

        x = self.features(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x,(1,1))

        x = torch.flatten(x,1)

        x = self.classifier(x)

        return x


# -----------------------------
# 计算参数量
# -----------------------------
def count_params(model):

    return sum(p.numel() for p in model.parameters())/1e6


# -----------------------------
# FLOPs
# -----------------------------
def compute_flops(model,device):

    input = torch.randn(1,3,224,224).to(device)

    macs, params = profile(
        model,
        inputs=(input,),
        verbose=False
    )

    return macs/1e9


# -----------------------------
# Latency
# -----------------------------
def compute_latency(model,device,runs=100):

    model.eval()

    input = torch.randn(1,3,224,224).to(device)

    with torch.no_grad():

        for _ in range(20):
            model(input)

        start = time.time()

        for _ in range(runs):
            model(input)

        end = time.time()

    latency = (end-start)/runs*1000

    fps = 1000/latency

    return latency,fps


# -----------------------------
# 读取最佳准确率
# -----------------------------
def read_best_acc(csv_path):

    df = pd.read_csv(csv_path)

    val_df = df[df.phase=="val"]

    return val_df.acc.max()


# -----------------------------
# 主函数
# -----------------------------
def main():

    os.makedirs("logs",exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results=[]


    # -----------------------------
    # MobileNetV2
    # -----------------------------
    model = models.mobilenet_v2(weights=None)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs,3)

    model.load_state_dict(
        torch.load("models/baseline_best.pth",map_location=device)
    )

    model=model.to(device)

    acc = read_best_acc("logs/baseline_log.csv")

    params = count_params(model)

    flops = compute_flops(model,device)

    latency,fps = compute_latency(model,device)

    results.append({
        "Model":"MobileNetV2",
        "Accuracy":round(acc,4),
        "Params(M)":round(params,2),
        "MAdds(B)":round(flops,2),
        "Latency(ms)":round(latency,2),
        "FPS":round(fps,1)
    })


    # -----------------------------
    # MobileNetV2-ECA
    # -----------------------------
    model = ImprovedMobileNetV2(num_classes=3)

    model.load_state_dict(
        torch.load("models/eca_best.pth",map_location=device)
    )

    model=model.to(device)

    acc = read_best_acc("logs/eca_log.csv")

    params = count_params(model)

    flops = compute_flops(model,device)

    latency,fps = compute_latency(model,device)

    results.append({
        "Model":"MobileNetV2-ECA",
        "Accuracy":round(acc,4),
        "Params(M)":round(params,2),
        "MAdds(B)":round(flops,2),
        "Latency(ms)":round(latency,2),
        "FPS":round(fps,1)
    })


    df = pd.DataFrame(results)

    save_path="logs/model_metrics.csv"

    df.to_csv(save_path,index=False)

    print("\n结果表格：")

    print(df)

    print("\n已保存到:",save_path)


if __name__=="__main__":

    main()