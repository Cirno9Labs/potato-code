import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. 解决 Windows 环境下的 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
        # x 形状: [batch_size, channels, h, w]
        b, c, _, _ = x.size()

        y = self.avg_pool(x)  # 形状变为: [b, c, 1, 1]

        # 优化点：使用 view 显式转换维度，替代脆弱的 squeeze/transpose 链条
        # Conv1d 期待的输入格式为: [batch_size, in_channels=1, seq_len=c]
        y = y.view(b, 1, c)

        y = self.conv(y)  # 经过 1D 卷积，输出形状仍为: [b, 1, c]

        # 将形状还原为与原特征图通道一致的权重向量
        y = y.view(b, c, 1, 1)
        y = self.sigmoid(y)

        # 将注意力权重按元素乘回原特征图
        return x * y.expand_as(x)


# --- 2. 定义改进后的模型架构 ---
class ImprovedMobileNetV3(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV3, self).__init__()
        # 加载官方预训练模型 (含原生 SE)
        base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # 提取主干特征网络
        self.features = base_model.features

        # 获取特征图输出通道数 (MobileNetV3-Large 最后通常是 960)
        last_channel = base_model.classifier[0].in_features

        # --- 外挂 ECA 模块 ---
        self.eca = ECAModule(last_channel)

        # 提取其余部分
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

        # 修改分类器最后一层为三分类 (Linear 层索引为 3)
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)  # 在特征图输出后立刻进行 ECA 加权
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- 3. 训练配置与主循环 ---
def main():
    # 数据路径
    DATA_DIR = r'/dataset'
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
    model = ImprovedMobileNetV3(num_classes=3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 改进版 MobileNetV3 + ECA 开始训练...")
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

    # 保存改进后的权重
    torch.save(model.state_dict(), 'improved_potato_mobilenetv3_eca.pth')
    print("\n✅ 改进模型训练完成，权重已保存至当前目录下的 improved_potato_mobilenetv3_eca.pth")


if __name__ == '__main__':
    main()