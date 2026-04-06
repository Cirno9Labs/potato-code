import os

# 解决 Windows 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from thop import profile


# ─── 1. 模块定义 (ECA & SE) ─────────────────────────────────────────────────

class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ─── 2. 模型封装定义 ───────────────────────────────────────────────────────

# MobileNetV2 + ECA (代码1核心模型)
class MobileNetV2_ECA(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = base.features
        for module in self.features:
            if isinstance(module, InvertedResidual):
                out_channels = module.out_channels
                module.conv.add_module('eca', ECAModule(out_channels))
        last_channel = base.classifier[1].in_features
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# MobileNetV2 + SE (代码2引入)
class MobileNetV2_SE(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = base.features
        for module in self.features:
            if isinstance(module, InvertedResidual):
                out_channels = module.out_channels
                module.conv.add_module('se', SEModule(out_channels))
        last_channel = base.classifier[1].in_features
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 基础模型获取函数
def get_model(name, num_classes=3):
    if name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─── 3. 评估核心逻辑 (保留代码1的严谨性) ─────────────────────────────────────

def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, p, r, f1, cm


def evaluate_hardware(model, device):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)

    # 延迟测试
    for _ in range(50): _ = model(dummy)  # 预热
    if device.type == 'cuda': torch.cuda.synchronize()

    start = time.time()
    for _ in range(200): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    latency = (time.time() - start) / 200 * 1000

    return params / 1e6, macs / 1e9, latency


# ─── 4. 主流程 ───────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_DIR = r'E:\Source\potato\dataset\Validation'

    val_loader = DataLoader(
        datasets.ImageFolder(VAL_DIR, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=32, shuffle=False, num_workers=4
    )

    # 模型注册表 (合并代码1与代码2)
    models_to_eval = {
        "MobileNetV2": {"model": get_model("MobileNetV2"), "path": "models/baseline_best.pth"},
        "MobileNetV2-ECA": {"model": MobileNetV2_ECA(), "path": "models/best_mobilenetv2_eca.pth"},
        "MobileNetV2-SE": {"model": MobileNetV2_SE(), "path": "models/se_best.pth"},
        "VGG16": {"model": get_model("VGG16"), "path": "models/vgg_aug_best.pth"},
        "ResNet50": {"model": get_model("ResNet50"), "path": "models/resnet_aug_best.pth"},
    }

    results = []
    for name, info in models_to_eval.items():
        print(f"🚀 正在评估: {name}...")
        model = info["model"].to(device)

        if os.path.exists(info["path"]):
            model.load_state_dict(torch.load(info["path"], map_location=device, weights_only=True))
        else:
            print(f"⚠️ 找不到权重 {info['path']}，跳过分类指标测试。")

        acc, p, r, f1, cm = evaluate_classification(model, val_loader, device)
        p_m, m_b, lat = evaluate_hardware(model, device)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro-P": round(p, 4),
            "Macro-R": round(r, 4),
            "Macro-F1": round(f1, 4),
            "Params(M)": round(p_m, 2),
            "MAdds(B)": round(m_b, 3),
            "Latency(ms)": round(lat, 2),
            "FPS": round(1000 / lat, 1)
        })
        print(f"Confusion Matrix for {name}:\n{cm}\n")

    # 保存 CSV (参考代码2逻辑)
    df = pd.DataFrame(results)
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/model_comparison_results.csv", index=False)

    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"✅ 所有对比数据已保存至: logs/model_comparison_results.csv")


if __name__ == "__main__":
    main()