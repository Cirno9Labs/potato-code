import os
import math
import csv

# 解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader


# ─── SE (Squeeze-and-Excitation) 模块 ───────────────────────────────────────────
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 模块（经典版本）
    reduction_ratio 通常取 16
    """
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        reduced_channels = max(channels // reduction, 1)  # 至少 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)          # [B, C]
        y = self.fc(y).view(b, c, 1, 1)          # [B, C, 1, 1]
        return x * y                             # 通道缩放


# ─── 加入 SE 的 MobileNetV2 ────────────────────────────────────────────────────
class MobileNetV2WithSE(nn.Module):
    def __init__(self, num_classes=3, reduction=16):
        super(MobileNetV2WithSE, self).__init__()
        # 载入预训练 MobileNetV2
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features

        # 遍历所有 InvertedResidual 块，在合适位置插入 SE
        for name, module in self.features.named_modules():
            if isinstance(module, InvertedResidual):
                # 通常在 expansion (pointwise 升维) 之后插入 SE 效果最好
                # module.conv 是一个 nn.Sequential，结构一般为：
                #   0: pointwise 1x1 升维
                #   1: BN
                #   2: ReLU6
                #   3: depthwise 3x3
                #   4: BN
                #   5: ReLU6
                #   6: pointwise 1x1 降维
                #   7: BN
                # 我们插入在 index 6 (降维 pointwise) 之后、残差相加之前

                out_channels = module.out_channels
                # 在 conv Sequential 最后插入 SE
                module.conv.add_module('se', SEModule(out_channels, reduction))

        # 分类头
        last_channel = base.classifier[1].in_features  # 1280
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ─── 随机选一种数据增强策略（与你之前代码一致） ───────────────────────────────
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
            tensor = transforms.ToTensor()(img)
            noise = torch.randn(tensor.size()) * self.noise_std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return transforms.ToPILImage()(tensor)   # 转回 PIL 以兼容后续 transforms


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_path = "logs/se_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    best_acc = 0.0

    # 资料路径（与你之前相同）
    DATA_DIR = r'/dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 统一转 Tensor
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

    # 初始化 MobileNetV2 + SE
    model = MobileNetV2WithSE(num_classes=3, reduction=16).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 可选：加入余弦退火（与 ECA 版本一致）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"🚀 MobileNetV2 + SE (reduction=16) 开始训练...")
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

            # 训练阶段结束后更新学习率
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'Epoch {epoch + 1}/{EPOCHS} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            csv_writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item()])

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "models/se_best.pth")
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")

    torch.save(model.state_dict(), "models/se_last.pth")
    log_file.close()
    print("\n✅ 训练结束！最佳模型已保存至 models/se_best.pth")


if __name__ == '__main__':
    main()