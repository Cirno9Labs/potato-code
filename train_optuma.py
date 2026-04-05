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
import optuna


# ─── ECA 注意力模块 ────────────────────────────────────────────────────────────
class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class ImprovedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedMobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        for module in self.features:
            if isinstance(module, InvertedResidual):
                out_channels = module.out_channels
                module.conv.add_module('eca', ECAModule(out_channels))
        last_channel = base.classifier[1].in_features
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
            return transforms.ToPILImage()(tensor)


# ─── Optuna 目标函数（核心调参部分） ─────────────────────────────────────────────
def objective(trial):
    # ─── 可调超参数（扩充搜索空间） ─────────────────────────────────────
    LR = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32])
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    OPTIMIZER_NAME = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')

    EPOCHS = 8

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomOneOfStrategy(brightness=0.35, noise_std=0.025),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
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

    # 修改 num_workers=4 提升数据加载速度 (Windows 下需在 __main__ 保护内运行，当前代码已满足)
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                            pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = [1303, 816, 1132]
    total = sum(class_counts)
    class_weights = torch.tensor([total / x for x in class_counts], dtype=torch.float).to(device)

    model = ImprovedMobileNetV2(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 动态选择优化器并加入权重衰减
    if OPTIMIZER_NAME == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 修复 FutureWarning，使用最新的 AMP API
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_acc = 0.0

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
                    if phase == 'train' and scaler is not None:
                        # 修复 FutureWarning，使用最新的 autocast API
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == 'val':
                trial.report(epoch_acc.item(), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if epoch_acc > best_acc:
                    best_acc = epoch_acc

    return best_acc


# ─── 启动 Optuna 调参 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    print("🚀 开始 Optuna 超参数优化（6GB RTX 3060 Laptop 优化版 v2）...")
    print("建议跑 15~20 次试验（n_trials=20）")
    study.optimize(objective, n_trials=20)

    print("\n✅ 优化完成！")
    print("最优超参数：")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print(f"最佳验证准确率: {study.best_value:.4f}")

    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))