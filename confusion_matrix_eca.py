import os
# 1. 解决 Windows OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader




# ─── ECA 模块定义（必须与训练代码中的安全维度变换版完全一致） ──────────────
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


# ─── 深度融合版模型类（必须匹配重构后的训练代码） ──────────────────────────
class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        # weights=None 因为我们稍后会手动加载训练好的 .pth
        base = models.mobilenet_v2(weights=None)
        self.features = base.features

        # 核心重构逻辑：将 ECA 注入每一个倒残差块
        for module in self.features:
            if isinstance(module, InvertedResidual):
                out_channels = module.out_channels
                module.conv.add_module('eca', ECAModule(out_channels))

        last_channel = base.classifier[1].in_features  # 1280
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def generate_confusion_matrix():
    # 路径配置
    BASE_DIR = r"E:\Source\potato"
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mobilenetv2_eca.pth")
    VAL_DATA_DIR = os.path.join(BASE_DIR, "dataset", "Validation")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：在 {MODEL_PATH} 找不到权重文件，请确认训练已完成。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证集预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 实例化模型并加载权重
    model = ImprovedMobileNetV2(num_classes=3)
    # 使用 weights_only=True 是更安全的加载方式
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✅ 深度融合版 ECA 模型载入成功！")
    print(f"类别映射：{dataset.class_to_idx}")

    y_true = []
    y_pred = []

    print("正在处理验证集数据...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘图设置
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)

    # 使用官方推荐的 plot 方式，并自定义颜色和格式
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Greens, values_format='d', colorbar=True)

    plt.title("Confusion Matrix: MobileNetV2 + Deep ECA (Potato Diseases)", fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # 保存结果
    os.makedirs("figures", exist_ok=True)
    save_path = "figures/confusion_matrix_eca_deep.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # 在控制台或 Notebook 中预览

    print(f"📊 混淆矩阵已保存至：{save_path}")


if __name__ == '__main__':
    generate_confusion_matrix()