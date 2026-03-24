import os

# 解决 Windows libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


# -----------------------------
# 数据增强策略
# -----------------------------
class RandomOneOfStrategy(object):

    def __init__(self, brightness=0.35, noise_std=0.025):
        self.brightness = brightness
        self.noise_std = noise_std

    def __call__(self, img):

        choice = torch.randint(0, 4, (1,)).item()

        if choice == 0:
            return img

        elif choice == 1:
            aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 2:
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 3:
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return noisy_tensor

        return img


# -----------------------------
# PR 曲线绘制函数
# -----------------------------
def plot_pr_curve(y_true, y_score, num_classes):

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(6,5), dpi=300)

    for i in range(num_classes):

        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            y_score[:, i]
        )

        ap = average_precision_score(
            y_true_bin[:, i],
            y_score[:, i]
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f'Class {i} (AP={ap:.3f})'
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim([0,1])
    plt.ylim([0.7,1.0])

    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)

    plt.savefig("figures/pr_curve.png", dpi=300)

    plt.show()


def main():

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_path = "logs/baseline_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # -----------------------------
    # 数据增强
    # -----------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: transforms.ToTensor()(x) if not isinstance(x, torch.Tensor) else x),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    # -----------------------------
    # 数据集
    # -----------------------------
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=train_transforms),
        'val': datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    }

    dataloaders = {

        'train': DataLoader(
            image_datasets['train'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        ),

        'val': DataLoader(
            image_datasets['val'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 类别权重
    # -----------------------------
    class_counts = [1303, 816, 1132]
    total = sum(class_counts)

    class_weights = torch.tensor(
        [total/x for x in class_counts],
        dtype=torch.float
    ).to(device)

    print("类别权重:", class_weights)

    # -----------------------------
    # 模型
    # -----------------------------
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3)

    model = model.to(device)

    # -----------------------------
    # Loss + Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 开始训练 MobileNetV2")

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(EPOCHS):

        for phase in ['train','val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)

                    loss = criterion(outputs,labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss/len(image_datasets[phase])
            epoch_acc = running_corrects.double()/len(image_datasets[phase])

            print(f'Epoch {epoch+1}/{EPOCHS} [{phase}] Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f}')

            csv_writer.writerow([epoch+1,phase,epoch_loss,epoch_acc.item()])

            if phase=='val' and epoch_acc>best_acc:

                best_acc = epoch_acc

                torch.save(
                    model.state_dict(),
                    "models/baseline_best.pth"
                )

                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    torch.save(
        model.state_dict(),
        "models/baseline_last.pth"
    )

    log_file.close()

    print("训练完成")

    # -----------------------------
    # 计算 PR 曲线
    # -----------------------------
    print("📊 计算 PR 曲线...")

    model.eval()

    y_true = []
    y_score = []

    with torch.no_grad():

        for inputs, labels in dataloaders['val']:

            inputs = inputs.to(device)

            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)

            y_score.append(probs.cpu().numpy())
            y_true.append(labels.numpy())

    y_score = np.concatenate(y_score)
    y_true = np.concatenate(y_true)

    plot_pr_curve(y_true, y_score, num_classes=3)


if __name__ == '__main__':

    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR,'Training')
    VAL_DIR = os.path.join(DATA_DIR,'Validation')

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    main()