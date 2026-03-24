import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# 读取日志
baseline = pd.read_csv("logs/baseline_log.csv")
eca = pd.read_csv("logs/eca_log.csv")
se = pd.read_csv("logs/se_log.csv")

# 训练集
train_base = baseline[baseline.phase=="train"]
train_eca = eca[eca.phase=="train"]
train_se = se[se.phase=="train"]

# 验证集
val_base = baseline[baseline.phase=="val"]
val_eca = eca[eca.phase=="val"]
val_se = se[se.phase=="val"]


# ===============================
# 1 训练集 Loss 曲线
# ===============================
plt.figure(figsize=(6,5), dpi=300)

plt.plot(train_base.epoch, train_base.loss, label="Baseline")
plt.plot(train_eca.epoch, train_eca.loss, label="ECA")
plt.plot(train_se.epoch, train_se.loss, label="SE")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("figures/train_loss_curve.png")
plt.close()


# ===============================
# 2 验证集 Loss 曲线
# ===============================
plt.figure(figsize=(6,5), dpi=300)

plt.plot(val_base.epoch, val_base.loss, label="Baseline")
plt.plot(val_eca.epoch, val_eca.loss, label="ECA")
plt.plot(val_se.epoch, val_se.loss, label="SE")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("figures/val_loss_curve.png")
plt.close()


# ===============================
# 3 验证集 Accuracy 曲线
# ===============================
plt.figure(figsize=(6,5), dpi=300)

plt.plot(val_base.epoch, val_base.acc, label="Baseline")
plt.plot(val_eca.epoch, val_eca.acc, label="ECA")
plt.plot(val_se.epoch, val_se.acc, label="SE")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("figures/val_accuracy_curve.png")
plt.close()


print("✅ 三张曲线图已保存到 figures/")