import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建保存图片的目录
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# 1. 读取日志 (请确保文件名与你训练保存的文件名一致)
# ---------------------------------------------------------
try:
    # 基础模型（带增强与不带增强）
    base_aug = pd.read_csv("logs/baseline_log.csv")  # MobileNetV2 + Aug
    # base_no_aug = pd.read_csv("logs/baseline_no_aug_log.csv")  # MobileNetV2 原始

    # 进阶架构（带增强）
    resnet = pd.read_csv("logs/resnet_aug_log.csv")
    vgg = pd.read_csv("logs/vgg_aug_log.csv")
    eca = pd.read_csv("logs/eca_log.csv")  # MobileNetV2 + ECA + Aug
except FileNotFoundError as e:
    print(f"❌ 错误：找不到日志文件。请检查文件名是否正确。\n具体信息：{e}")
    exit()

# 准备绘图配置：模型对象、标签、颜色
model_configs = [
    # {"df": base_no_aug, "label": "MobileNetV2 (No Aug)", "color": "gray", "ls": "--"},
    {"df": base_aug, "label": "MobileNetV2", "color": "blue", "ls": "-"},
    {"df": resnet, "label": "ResNet50", "color": "green", "ls": "-"},
    {"df": vgg, "label": "VGG16", "color": "orange", "ls": "-"},
    {"df": eca, "label": "MobileNetV2-ECA", "color": "red", "ls": "-"},
]


# =========================================================
# 绘图函数：减少重复代码
# =========================================================
def plot_curves(phase, metric, title, ylabel, filename):
    plt.figure(figsize=(8, 6), dpi=300)

    for cfg in model_configs:
        data = cfg["df"][cfg["df"].phase == phase]
        plt.plot(
            data.epoch,
            data[metric],
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.5
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()
    print(f"✅ 已生成: {filename}")


# =========================================================
# 生成三幅核心对比图
# =========================================================

# 1. 训练集 Loss 曲线 (看收敛速度)
plot_curves("train", "loss", "Training Loss Comparison", "Loss", "comparison_train_loss.png")

# 2. 验证集 Loss 曲线 (看是否过拟合)
plot_curves("val", "loss", "Validation Loss Comparison", "Loss", "comparison_val_loss.png")

# 3. 验证集 Accuracy 曲线 (看最终精度)
plot_curves("val", "acc", "Validation Accuracy Comparison", "Accuracy", "comparison_val_acc.png")

print("\n🚀 所有对比分析图已保存至 figures/ 文件夹。")