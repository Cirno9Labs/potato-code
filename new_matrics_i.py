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
import matplotlib.pyplot as plt  # 用于生成图片样式的表格（高清PNG）

# ─── 1. 模块定义 (ECA) ─────────────────────────────────────────────────────
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


# ─── 2. MobileNetV2-ECA 模型定义 ───────────────────────────────────────────
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


# ─── 3. 评估核心逻辑（新增三个类别的 per-class Accuracy） ─────────────────
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

    # 总体准确率
    acc = accuracy_score(all_labels, all_preds)

    # Macro 平均指标
    p_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 每类指标（早疫病、健康、晚疫病）
    p_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
    r_per = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # 新增：计算每类 Accuracy（One-vs-Rest 方式：(TP + TN) / Total）
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        acc_i = (tp + tn) / total if total > 0 else 0.0
        per_class_acc.append(acc_i)

    return acc, p_macro, r_macro, f1_macro, p_per, r_per, f1_per, per_class_acc


def evaluate_hardware(model, device):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)

    # 延迟测试（预热 + 正式测试）
    for _ in range(50):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(200):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency = (time.time() - start) / 200 * 1000

    return params / 1e6, macs / 1e9, latency


# ─── 4. 主流程（仅生成 MobileNetV2-ECA 的表格 + 图片风格 PNG） ─────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================== 请根据实际情况修改 ==================
    VAL_DIR = r'E:\Source\potato\dataset\Validation'          # 验证集路径
    # ======================================================

    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    print(f"✅ 数据集发现类别: {val_dataset.classes}")
    print("   提示：ImageFolder 按文件夹名称字母顺序排序，对应 [早疫病, 健康, 晚疫病]")

    # 参考文献中定义的三类中文标签
    class_labels = ['早疫病', '健康', '晚疫病']

    # 仅评估 MobileNetV2-ECA
    models_to_eval = {
        "MobileNetV2-ECA": {"model": MobileNetV2_ECA(), "path": "models/best_mobilenetv2_eca.pth"},
    }

    results = []
    for name, info in models_to_eval.items():
        print(f"🚀 正在评估: {name}...")

        model = info["model"].to(device)

        if os.path.exists(info["path"]):
            model.load_state_dict(torch.load(info["path"], map_location=device, weights_only=True))
            print(f"   已加载权重: {info['path']}")
        else:
            print(f"⚠️ 找不到权重 {info['path']}，请检查路径！")
            continue

        # 分类性能评估（新增 per-class ACC）
        acc, p_m, r_m, f1_m, p_per, r_per, f1_per, per_acc = evaluate_classification(model, val_loader, device)

        # 模型复杂度与实时性评估
        p_count, m_b, lat = evaluate_hardware(model, device)

        # 构建结果字典（新增 早疫病-ACC、健康-ACC、晚疫病-ACC）
        result = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro-P": round(p_m, 4),
            "Macro-R": round(r_m, 4),
            "Macro-F1": round(f1_m, 4),
        }

        for i, label in enumerate(class_labels):
            if i < len(p_per):
                result[f"{label}-ACC"] = round(per_acc[i], 4)
                result[f"{label}-P"] = round(p_per[i], 4)
                result[f"{label}-R"] = round(r_per[i], 4)
                result[f"{label}-F1"] = round(f1_per[i], 4)

        # 硬件指标
        result.update({
            "Params(M)": round(p_count, 2),
            "MAdds(B)": round(m_b, 3),
            "Latency(ms)": round(lat, 2),
            "FPS": round(1000 / lat, 1) if lat > 0 else 0
        })

        results.append(result)

    # 保存 CSV 表格
    df = pd.DataFrame(results)
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/mobilenetv2_eca_performance.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ─── 生成图片样式的表格（高清PNG，适合直接插入论文） ───
    fig = plt.figure(figsize=(26, 8))   # 增加宽度以容纳新增的 -ACC 列
    ax = fig.add_subplot(111)
    ax.axis('off')

    # 表格内容
    table = ax.table(
        cellText=df.round(4).astype(str).values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.095] * len(df.columns)   # 适当缩小列宽以适应更多列
    )

    # 美化图片风格（蓝色表头 + 清晰边框 + 大字体）
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.35, 2.8)
    for key, cell in table._cells.items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        if key[0] == 0:  # 表头
            cell.set_facecolor('#1E90FF')      # 蓝色表头
            cell.set_text_props(weight='bold', color='white')

    plt.title('MobileNetV2-ECA 马铃薯病害分类模型性能表\n'
              '(早疫病 / 健康 / 晚疫病 的 ACC / P / R / F1 + Macro 平均 + 模型复杂度与实时性)',
              fontsize=18, pad=30, fontweight='bold', color='#1E90FF')

    png_path = "logs/mobilenetv2_eca_performance_table.png"
    plt.savefig(png_path, dpi=500, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    # ─── 终端输出 ───
    print("=" * 140)
    print(df.to_string(index=False))
    print("=" * 140)
    print(f"✅ CSV表格已保存至: {csv_path}")
    print(f"✅ 图片样式的表格（高清PNG）已保存至: {png_path}  ← 可直接插入论文/报告")
    print("   已严格按照参考文献要求实现全部指标，并新增三个类别的 per-class Accuracy（ACC）。")
    print("   （早疫病-ACC、健康-ACC、晚疫病-ACC 已使用 One-vs-Rest 方式计算完成）")


if __name__ == "__main__":
    main()