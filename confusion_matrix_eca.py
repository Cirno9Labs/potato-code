import os
# 1. 必须在 import torch 之前设置，解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import math

# 1. 解決 Windows OpenMP 衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── ECA 模塊定義（必須與訓練時完全相同） ────────────────────────────────
class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y).expand_as(x)

# ─── 訓練時用的自訂模型（必須完全一樣） ────────────────────────────────────
class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights=None)  # 不載預訓練，避免衝突
        self.features = base.features
        last_channel = base.classifier[1].in_features  # 1280
        self.eca = ECAModule(last_channel)
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ─── 主程式 ──────────────────────────────────────────────────────────────────
BASE_DIR = r"E:\Source\potato"
MODEL_PATH = os.path.join(BASE_DIR, "models", "eca_best.pth")
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Validation")

if not os.path.exists(MODEL_PATH):
    print(f"錯誤：找不到模型 {MODEL_PATH}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# 使用自訂模型載入 ECA 權重
model = ImprovedMobileNetV2(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

print(f"ECA 模型載入成功，類別：{dataset.classes}")

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 混淆矩陣 - 使用更美觀的 ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)

fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

plt.title("Confusion Matrix - ECA Best Model")
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
save_path = "figures/confusion_matrix_eca.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ ECA 混淆矩陣已保存至：{save_path}")