import os
import math

# 1. 解决 Windows 环境下的 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import csv

# --- 1. 定义 ECA 注意力模块 ---
class ECAModule(nn.Module):
    """
    ECA (Efficient Channel Attention) 模块
    通过自适应核大小的 1D 卷积实现跨通道交互，不降维，极轻量。
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        # 根据通道数自适应计算 1D 卷积核大小 k
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, 1, c)
        y = self.conv(y)
        y = y.view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# --- 2. 定义改进后的 MobileNetV2 模型架构 ---
class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        # 加载官方预训练模型
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # 提取主干特征网络
        self.features = base_model.features

        # 获取特征图输出通道数 (MobileNetV2 最后通常是 1280)
        last_channel = base_model.classifier[1].in_features

        # --- 外挂 ECA 模块 ---
        self.eca = ECAModule(last_channel)

        # 提取分类器部分，并修改为三分类 (Linear 层索引为 1)
        self.classifier = base_model.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)  # 在特征提取后立刻进行 ECA 加权

        # MobileNetV2 的官方实现中没有单独的 avgpool 属性，通常用 F.adaptive_avg_pool2d 代替
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- 3. 训练配置与主循环 ---
def main():
    # CSV日志
    log_path = "logs/eca_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "phase", "loss", "acc"])

    # Best model记录
    best_acc = 0.0
    # 数据路径
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.0003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    # 初始化改进模型
    model = ImprovedMobileNetV2(num_classes=3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 改进版 MobileNetV2 + ECA 开始训练...")
    print(f"训练设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"训练样本数: {len(image_datasets['train'])}")
    print(f"验证样本数: {len(image_datasets['val'])}\n")

    # 训练循环
    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
                torch.save(model.state_dict(), "models/baseline_best.pth")
                print(f"⭐ Best model updated! Acc={best_acc:.4f}")


    # 保存训练好的最佳权重
    torch.save(model.state_dict(), 'models/eca_best.pth')
    log_file.close()
    print("\n✅ 训练成功结束！最佳权重已保存至models/eca_best.pth")

if __name__ == '__main__':
    main()