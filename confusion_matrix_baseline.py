import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 解決 Windows libiomp5md.dll 衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = r'E:\Source\potato\dataset\Validation'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 建立與訓練時相同的模型結構
model = models.mobilenet_v2(weights=None)  # 先不載預訓練，避免權重衝突
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 3)

# 載入權重（建議加上 weights_only=True 避免警告）
model_path = "models/baseline_best.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 計算混淆矩陣
cm = confusion_matrix(y_true, y_pred)

# 使用 ConfusionMatrixDisplay 以藍色為主色調繪製
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

fig, ax = plt.subplots(figsize=(7, 7))  # 稍微大一點更好看
disp.plot(
    ax=ax,
    cmap='Blues',           # 藍色主題（從淺到深）
    colorbar=True,          # 顯示右側顏色條
    values_format='d'       # 顯示整數
)

# 調整文字顏色（數字在深藍處會自動變白，在淺藍處變黑）
for text in ax.texts:
    text.set_color('black' if int(text.get_text()) < cm.max() / 2 else 'white')

plt.title("Confusion Matrix - Baseline Model", fontsize=14, pad=20)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)

os.makedirs("figures", exist_ok=True)
save_path = "figures/confusion_matrix_baseline_blue.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 藍色主題的 baseline 混淆矩陣已保存至：{save_path}")