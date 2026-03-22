import os
import math
import csv

# 1. 解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader


# ─── ECA 注意力模块 ────────────────────────────────────────────────────────────
class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块
    使用自适应 1D 卷积核大小，更安全的维度变换写法
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # 保证奇数 kernel size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)

        # 安全的维度变换：[B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)

        # 变换回原始维度：[B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * self.sigmoid(y).expand_as(x)


# ─── 加入 ECA 的 MobileNetV2 (深度融合版) ──────────────────────────────────────
class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        # 载入预训练权重
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features

        # 遍历所有的层，将 ECA 模块无缝嵌入到每一个 InvertedResidual 块中
        for module in self.features:
            if isinstance(module, InvertedResidual):
                # InvertedResidual 的 conv 属性是一个 nn.Sequential
                # 我们将 ECA 挂载在这个 Sequential 的最后
                # 这样它就会作用于残差相加之前的特征图上
                out_channels = module.out_channels
                module.conv.add_module('eca', ECAModule(out_channels))

        # 替换分类头
        last_channel = base.classifier[1].in_features  # 1280
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ─── 自定义：随机选一种增强策略 ────────────────────────────────────────────────
class RandomOneOfStrategy(object):
    def __init__(self, brightness=0.35, noise_std=0.025):
        self.brightness = brightness
        self.noise_std = noise_std

    def __call__(self, img):
        choice = torch.randint(0, 4, (1,)).item()

        if choice == 0:
            return img
        elif choice == 1:
            return transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])(img)
        elif choice == 2:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=self.brightness)
            ])(img)
        elif choice == 3:
            # 兼容处理：确保送入 CenterCrop 前的数据格式正确
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return transforms.ToPILImage()(tensor)

        # ─── 主训练函数 ────────────────────────────────────────────────────────────────


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_path = "logs/eca_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # 资料路径
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 统一转换为 Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=train_transforms),
        'val': datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = [1303, 816, 1132]
    total = sum(class_counts)
    class_weights = torch.tensor([total / x for x in class_counts], dtype=torch.float).to(device)
    print("类别权重:", class_weights)

    # 初始化模型
    model = ImprovedMobileNetV2(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 增加余弦退火学习率调度器 (极力推荐)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"🚀 MobileNetV2 + 深度融合 ECA 开始训练...")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"训练样本数: {len(image_datasets['train'])}")
    print(f"验证样本数: {len(image_datasets['val'])}\n")

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

            # 只有在训练阶段结束后，才更新学习率
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'Epoch {epoch + 1}/{EPOCHS} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            csv_writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item()])

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "models/eca_best.pth")
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    torch.save(model.state_dict(), "models/eca_last.pth")
    log_file.close()
    print("\n✅ 训练结束！最佳模型已保存至 models/eca_best.pth")


if __name__ == '__main__':
    main()