import os

# 1. 必须在 import torch 之前设置，解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = r'E:\Source\potato\dataset\Validation'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs,3)

model.load_state_dict(torch.load("models/potato_mobilenetv2_baseline_last.pth"))

model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs,labels in loader:

        inputs = inputs.to(device)

        outputs = model(inputs)
        _,preds = torch.max(outputs,1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true,y_pred)

classes = dataset.classes

plt.figure(figsize=(6,6))
plt.imshow(cm)

plt.xticks(np.arange(len(classes)),classes)
plt.yticks(np.arange(len(classes)),classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j,i,cm[i,j],ha="center",va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

os.makedirs("figures",exist_ok=True)

plt.savefig("figures/confusion_matrix_baseline.png")


print("✅ 混淆矩阵已保存")