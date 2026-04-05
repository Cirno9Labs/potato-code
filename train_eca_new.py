import os
# 1. 解决 Windows 下的 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader, ConcatDataset




# ─── ECA 注意力模块 (保持不变) ───────────────────────────────────────────────────
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


# ─── 最终完整训练流程 ────────────────────────────────────────────────────────
def train_final_model():
    # ─── 1. 应用 Optuna 搜索到的最优超参数 ───
    LR = 0.00044164553065255885
    BATCH_SIZE = 32
    WEIGHT_DECAY = 3.643526297879779e-05
    EPOCHS = 30  # 最终训练建议跑 30-50 个 Epoch，确保充分收敛

    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_mobilenetv2_eca.pth")

    # ─── 2. 资料路径设置 ───
    DATA_DIR = r'E:\Source\potato\dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    TEST_DIR = os.path.join(DATA_DIR, 'Testing')  # 将原测试集并入训练
    VAL_DIR = os.path.join(DATA_DIR, 'Validation')  # 用于检验

    # ─── 3. 数据增强与加载 ───
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

    # 加载各自的文件夹
    ds_train_orig = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    ds_test_orig = datasets.ImageFolder(TEST_DIR, transform=train_transforms)
    ds_val = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    # 合并 Training 和 Testing 成为新的训练集
    combined_train_dataset = ConcatDataset([ds_train_orig, ds_test_orig])

    dataloaders = {
        'train': DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                            pin_memory=True),
        'val': DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    # ─── 4. 动态计算合并后数据集的类别权重 ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")

    # 提取合并后所有样本的标签
    all_targets = ds_train_orig.targets + ds_test_orig.targets
    num_classes = len(ds_train_orig.classes)
    class_counts = [all_targets.count(i) for i in range(num_classes)]
    total_samples = sum(class_counts)

    # 计算权重（总样本数 / 各类别样本数），应对不平衡
    class_weights = torch.tensor([total_samples / x for x in class_counts], dtype=torch.float).to(device)
    print(f"📊 合并后各类别样本数: {class_counts}")
    print(f"⚖️ 重新计算的类别权重: {class_weights.tolist()}")

    # ─── 5. 模型、损失函数、优化器初始化 ───
    model = ImprovedMobileNetV2(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 使用最优优化器 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ─── 6. 开始训练 ───
    best_acc = 0.0

    print("🚀 开始最终训练流程...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataset_size = len(combined_train_dataset)
            else:
                model.eval()
                dataset_size = len(ds_val)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and scaler is not None:
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

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"   {phase.capitalize()} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 保存最佳验证集模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"   🌟 发现更优模型！已保存至 {best_model_path}")

    print("\n✅ 训练完成！")
    print(f"🏆 最高验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    train_final_model()