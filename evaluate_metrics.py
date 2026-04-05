import os
# 解决 Windows 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from thop import profile



# ─── 1. 模型结构定义 ────────────────────────────────────────────────────────
# ECA 模块与改进的 MobileNetV2
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


class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights=None)  # 评估时无需预训练权重
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


def get_baseline_model(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


# ─── 2. 评估函数：分类指标 ──────────────────────────────────────────────────
def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算各项指标 (Macro 平均)，对应论文公式 (2-10)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, cm


# ─── 3. 评估函数：硬件与实时性指标 ──────────────────────────────────────────
def evaluate_hardware_metrics(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 计算 Params 和 MAdds(FLOPs)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    params_m = params / 1e6  # 转换为 Million (M)
    madds_b = macs / 1e9  # 转换为 Billion (B)

    # 计算推理延迟 (Latency)
    # GPU 预热
    for _ in range(50):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    num_iterations = 300
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    latency_ms = (total_time / num_iterations) * 1000

    return params_m, madds_b, latency_ms


# ─── 4. 主流程代码 ──────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用评估设备: {device}")

    # --- 资料路径与数据加载 ---
    VAL_DIR = r'E:\Source\potato\dataset\Validation'
    BATCH_SIZE = 32

    # 此处你可以根据上一轮的建议，加入高斯模糊或亮度调整来增加验证集难度
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ 验证集加载成功，共 {len(val_dataset)} 张图片。")

    # --- 模型注册 ---
    models_dict = {
        "MobileNetV2": {
            "model": get_baseline_model(num_classes=3).to(device),
            "weight_path": "models/baseline_best.pth"
        },
        "MobileNetV2-ECA": {
            "model": ImprovedMobileNetV2(num_classes=3).to(device),
            "weight_path": "models/best_mobilenetv2_eca.pth"
        }
    }

    results = []

    print("\n🚀 开始评估...")
    for model_name, info in models_dict.items():
        print(f"\n[{model_name}] 正在评估...")
        model = info["model"]
        weight_path = info["weight_path"]

        # 加载权重 (加入 weights_only=True 解决警告)
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        else:
            print(f"⚠️ 警告: 找不到权重文件 {weight_path}，将使用随机权重评估！")

        # 1. 测算分类指标
        acc, macro_p, macro_r, macro_f1, cm = evaluate_classification(model, val_loader, device)

        # 2. 测算硬件与实时指标 (移除了 FPS)
        params_m, madds_b, latency = evaluate_hardware_metrics(model, device)

        # 记录结果
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Macro-P": macro_p,
            "Macro-R": macro_r,
            "Macro-F1": macro_f1,
            "Params(M)": params_m,
            "MAdds(B)": madds_b,
            "Latency(ms)": latency
        })

        # 分开打印混淆矩阵
        print(f"[{model_name}] Confusion Matrix:")
        print(cm)

    # ─── 5. 打印对比表格 (完全贴合论文要求的指标) ─────────────────────────
    print("\n\n" + "=" * 95)
    print(" 综合评价指标对比表格 (Classification & Performance Metrics)")
    print("=" * 95)

    # 更新表头，加入 Macro-P 和 Macro-R，移除 FPS
    header = f"{'Model':<18} | {'Accuracy':<8} | {'Macro-P':<8} | {'Macro-R':<8} | {'Macro-F1':<8} | {'Params(M)':<9} | {'MAdds(B)':<8} | {'Latency(ms)':<11}"
    print(header)
    print("-" * 95)

    # 打印每一行结果
    for res in results:
        row = (f"{res['Model']:<18} | "
               f"{res['Accuracy']:.4f}   | "
               f"{res['Macro-P']:.4f}   | "
               f"{res['Macro-R']:.4f}   | "
               f"{res['Macro-F1']:.4f}   | "
               f"{res['Params(M)']:.2f}      | "
               f"{res['MAdds(B)']:.2f}     | "
               f"{res['Latency(ms)']:.2f}")
        print(row)
    print("=" * 95)


if __name__ == "__main__":
    main()