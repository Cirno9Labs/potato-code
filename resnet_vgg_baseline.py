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
# 自定义：每张图片随机且**互斥**地选一种增强策略
# -----------------------------
class RandomOneOfStrategy(object):
    def __init__(self, brightness=0.35, noise_std=0.025):
        self.brightness = brightness
        self.noise_std = noise_std

    def __call__(self, img):
        choice = torch.randint(0, 4, (1,)).item()  # 0~3 等概率随机选

        if choice == 0:  # 不增强
            return img

        elif choice == 1:  # 垂直翻转 + 亮度
            aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 2:  # 水平翻转 + 亮度
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])
            return aug(img)

        elif choice == 3:  # 高斯噪声策略
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return noisy_tensor

        return img


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 动态日志名称
    log_path = f"logs/{MODEL_NAME}_aug_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # -----------------------------
    # 1. 数据增强
    # -----------------------------
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
    # 3. 类别权重
    # -----------------------------
    class_counts = [1303, 816, 1132]
    total = sum(class_counts)
    class_weights = [total / x for x in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("类别权重:", class_weights)

    # -----------------------------
    # 4. 模型选择与初始化
    # -----------------------------
    if MODEL_NAME == 'resnet':
        # 使用 ResNet50 作为 baseline
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # 修改最后的全连接层

    elif MODEL_NAME == 'vgg':
        # 使用 VGG16 作为 baseline
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 3)  # 修改分类器的最后一层

    else:
        raise ValueError("不支持的模型名称，请选择 'resnet' 或 'vgg'")

    model = model.to(device)

    # -----------------------------
    # 5. Loss + Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 环境就绪！开始训练 {MODEL_NAME.upper()} (在线数据增强版本)...")
    print(f"使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"训练样本数: {len(image_datasets['train'])}")
    print(f"验证样本数: {len(image_datasets['val'])}\n")

    # -----------------------------
    # 6. 训练循环
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

            print(f'Epoch {epoch + 1}/{EPOCHS} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            csv_writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item()])

            # 保存best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(
                    model.state_dict(),
                    f"models/{MODEL_NAME}_aug_best.pth"
                )
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    torch.save(
        model.state_dict(),
        f"models/{MODEL_NAME}_aug_last.pth"
    )

    log_file.close()
    print(f"\n✅ {MODEL_NAME.upper()} 训练成功结束！")


if __name__ == '__main__':
    # --- 模型切换开关 ---
    # 在这里修改你想训练的模型：'resnet' 或 'vgg'
    # MODEL_NAME = 'resnet'
    MODEL_NAME = 'vgg '

    # --- 数据路径 ---
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    # --- 超参数 ---
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    # 💡 提示：VGG16 模型参数量很大，如果在训练 VGG 时遇到显存不足 (CUDA Out of Memory)
    # 请尝试将 BATCH_SIZE 调小（例如从 32 降到 16 或 8）。
    if MODEL_NAME == 'vgg':
        BATCH_SIZE = 16  # 为 VGG 提供一个默认的保守显存策略

    main()