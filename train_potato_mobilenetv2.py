import os

# 1. 必须在 import torch 之前设置，解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import csv

# -----------------------------
# 自定义：高斯噪声（tensor 专用）
# -----------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.025):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


# -----------------------------
# 自定义：每张图片随机且**互斥**地选一种增强策略（完美满足你的需求）
# 策略0：不增强
# 策略1：垂直翻转 + 亮度抖动
# 策略2：水平翻转 + 亮度抖动
# 策略3：高斯噪声（在 ToTensor 后自动处理）
# -----------------------------
class RandomOneOfStrategy(object):
    def __init__(self, brightness=0.35, noise_std=0.025):
        self.brightness = brightness
        self.noise_std = noise_std

    def __call__(self, img):
        # img 是 PIL Image
        choice = torch.randint(0, 4, (1,)).item()   # 0~3 等概率随机选

        if choice == 0:   # 不增强
            return img

        elif choice == 1:   # 垂直翻转 + 亮度
            aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 2:   # 水平翻转 + 亮度
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 3:   # 高斯噪声策略（这里转成 tensor 并加噪）
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)   # 防止越界
            return noisy_tensor

        return img


def main():

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # CSV日志
    log_path = "logs/baseline_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # -----------------------------
    # 1. 数据增强（已完美实现你的三种策略 + 不增强）
    # -----------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),   # ← 核心：随机选一种策略
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: transforms.ToTensor()(x) if not isinstance(x, torch.Tensor) else x),  # 处理高斯策略已返回 tensor 的情况
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # 验证集：完全不增强
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # 2. 数据加载
    # -----------------------------
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=train_transforms),
        'val': datasets.ImageFolder(VAL_DIR, transform=val_transforms)   # ← 修复：这里现在有定义了
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
    # 3. 类别权重 (解决类别不平衡)
    # -----------------------------
    class_counts = [1303, 816, 1132]
    total = sum(class_counts)
    class_weights = [total / x for x in class_counts]
    class_weights = torch.tensor(class_weights).to(device)
    print("类别权重:", class_weights)

    # -----------------------------
    # 4. 模型
    # -----------------------------
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    # -----------------------------
    # 5. Loss + Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 环境就绪！开始训练 MobileNetV2 Baseline...")
    print(f"使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"训练样本数: {len(image_datasets['train'])}")
    print(f"验证样本数: {len(image_datasets['val'])}\n")

    # -----------------------------
    # 6. 训练循环（保持原样）
    # -----------------------------
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

            print(f'Epoch {epoch+1}/{EPOCHS} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            csv_writer.writerow([epoch+1, phase, epoch_loss, epoch_acc.item()])

            # 保存best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(
                    model.state_dict(),
                    "models/baseline_best.pth"
                )
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    torch.save(
        model.state_dict(),
        "models/potato_mobilenetv2_baseline_last.pth"
    )

    log_file.close()
    print("\n✅ 训练成功结束！")


if __name__ == '__main__':
    # --- 数据路径 ---
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    # --- 超参数 ---
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    main()