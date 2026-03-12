import os

# 1. 必须在 import torch 之前设置，解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- 根据你的 tree 结果设置路径 ---
DATA_DIR = r'E:\Source\potato\dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
VAL_DIR = os.path.join(DATA_DIR, 'Validation')

# --- 超参数 ---
BATCH_SIZE = 32  # 你的 3060 6GB 显存，32 是最稳的
EPOCHS = 15  # 初始跑 15 轮看收敛情况
LR = 0.0003  # 较小的学习率适合迁移学习


def main():
    # 2. 图像预处理
    # MobileNetV3 论文指出，224x224 是性能与延迟的最佳平衡点
    data_transforms = {
        'Training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. 加载数据集
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['Training']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['Validation'])
    }

    # Windows 环境下 num_workers 建议设为 0，否则容易报多线程读取错误
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 初始化 MobileNetV3-Large 并使用预训练权重
    # MobileNetV3 核心改进：引入了 h-swish 激活函数和 SE 注意力机制

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # 修改最后的全连接层（classifier 里的最后一层索引是 3）
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 3)  # 三分类：Early, Late, Healthy

    model = model.to(device)

    # 5. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 环境就绪！开始训练...")
    print(f"使用设备: {torch.cuda.get_device_name(0)}")
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
                inputs, labels = inputs.to(device), labels.to(device)
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

    # 保存训练好的权重
    torch.save(model.state_dict(), 'potato_mobilenetv3_baseline.pth')
    print("\n✅ 训练成功结束！权重已保存至当前目录下的 potato_mobilenetv3_baseline.pth")


if __name__ == '__main__':
    main()