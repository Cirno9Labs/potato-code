import os
import math

# 1. 解決 Windows 下的 libiomp5md.dll 衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import csv


# ─── ECA 注意力模塊 ────────────────────────────────────────────────────────────
class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) 模塊
    極輕量級通道注意力，不降維，使用自適應 1D 卷積核大小
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # 保證奇數 kernel size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y).expand_as(x)


# ─── 加入 ECA 的 MobileNetV2 ───────────────────────────────────────────────────
class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.features = base.features
        last_channel = base.classifier[1].in_features  # 通常是 1280

        self.eca = ECAModule(last_channel)

        # 保留 classifier 結構，只替換最後一層 Linear
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)  # 在最後一組特徵圖上加 ECA
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ─── 自定義：隨機選一種增強策略（包含不增強） ────────────────────────────────
class RandomOneOfStrategy(object):
    def __init__(self, brightness=0.35, noise_std=0.025):
        self.brightness = brightness
        self.noise_std = noise_std

    def __call__(self, img):
        # img 是 PIL Image
        choice = torch.randint(0, 4, (1,)).item()  # 0,1,2,3 四選一

        if choice == 0:  # 不增強
            return img

        elif choice == 1:  # 垂直翻轉 + 亮度
            return transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])(img)

        elif choice == 2:  # 水平翻轉 + 亮度
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])(img)

        elif choice == 3:  # 只加高斯噪聲（轉 tensor 後處理）
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            return torch.clamp(tensor + noise, 0.0, 1.0)


# ─── 主訓練函數 ────────────────────────────────────────────────────────────────
def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # CSV 日誌
    log_path = "logs/eca_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # 資料路徑
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    # 超參數
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    # ─── 訓練資料增強（與 baseline 一致的隨機一種策略） ────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: transforms.ToTensor()(x) if not isinstance(x, torch.Tensor) else x),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # 驗證集：標準不增強
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # 資料集與載入器
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=train_transforms),
        'val': datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 類別權重（與 baseline 一致）
    class_counts = [1303, 816, 1132]
    total = sum(class_counts)
    class_weights = torch.tensor([total / x for x in class_counts], dtype=torch.float).to(device)
    print("類別權重:", class_weights)

    # 模型
    model = ImprovedMobileNetV2(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 MobileNetV2 + ECA 開始訓練...")
    print(f"設備: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"訓練樣本數: {len(image_datasets['train'])}")
    print(f"驗證樣本數: {len(image_datasets['val'])}\n")

    # 訓練迴圈
    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
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

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'Epoch {epoch + 1}/{EPOCHS} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            csv_writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item()])

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "models/eca_best.pth")
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    # 最終保存
    torch.save(model.state_dict(), "models/eca_last.pth")
    log_file.close()
    print("\n✅ 訓練結束！最佳模型已保存至 models/eca_best.pth")


if __name__ == '__main__':
    main()