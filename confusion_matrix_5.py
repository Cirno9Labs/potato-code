import os

# 解决 Windows 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader


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


def get_standard_model(name, num_classes=3):
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


# ─── 3. 主流程配置与执行 ───────────────────────────────────────────────────
def main():
    DATA_DIR = r'E:\Source\potato\dataset\Validation'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用计算设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    os.makedirs("figures", exist_ok=True)

    # 注册表：包含模型、权重路径和指定颜色 (红橙黄绿蓝)
    # cmap 采用 matplotlib 的标准 Sequential 渐变色系
    models_config = [
        {"name": "MobileNetV2", "model": get_standard_model("MobileNetV2"), "path": "models/baseline_best.pth",
         "color": "Reds"},
        {"name": "MobileNetV2-ECA", "model": MobileNetV2_ECA(), "path": "models/best_mobilenetv2_eca.pth",
         "color": "Oranges"},
        {"name": "MobileNetV2-SE", "model": MobileNetV2_SE(), "path": "models/se_best.pth", "color": "YlOrBr"},
        # 黄偏橙棕色，在白底上更易读
        {"name": "VGG16", "model": get_standard_model("VGG16"), "path": "models/vgg_aug_best.pth", "color": "Greens"},
        {"name": "ResNet50", "model": get_standard_model("ResNet50"), "path": "models/resnet_aug_best.pth",
         "color": "Blues"}
    ]

    print("\n🚀 开始生成混淆矩阵...")

    for config in models_config:
        model_name = config["name"]
        model = config["model"].to(device)
        weight_path = config["path"]
        cmap_color = config["color"]

        print(f"处理模型: {model_name} (颜色主题: {cmap_color})")

        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        else:
            print(f"⚠️ 警告：未找到权重文件 {weight_path}，当前输出图为随机预测结果！")

        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 绘图逻辑
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
        fig, ax = plt.subplots(figsize=(7, 7))

        disp.plot(
            ax=ax,
            cmap=cmap_color,
            colorbar=True,
            values_format='d'
        )

        # 动态调整字体颜色以保证对比度
        threshold = cm.max() / 2
        for text in ax.texts:
            val = int(text.get_text())
            text.set_color('white' if val > threshold else 'black')

        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=20)
        plt.xlabel("Predicted label", fontsize=12)
        plt.ylabel("True label", fontsize=12)

        # 保存图像
        save_path = f"figures/confusion_matrix_{model_name.lower().replace('-', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 图表已保存至: {save_path}\n")

    print("🎉 所有混淆矩阵已生成完毕！")


if __name__ == "__main__":
    main()