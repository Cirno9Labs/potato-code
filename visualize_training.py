import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

baseline = pd.read_csv("logs/baseline_log.csv")
eca = pd.read_csv("logs/eca_log.csv")
se = pd.read_csv("logs/se_log.csv")
# train_base = baseline[baseline.phase=="train"]
val_base = baseline[baseline.phase=="val"]

# train_eca = eca[eca.phase=="train"]
val_eca = eca[eca.phase=="val"]
val_se = se[se.phase=="val"]


# Loss Curve
plt.figure()

# plt.plot(train_base.epoch, train_base.loss,label="Baseline Train")
plt.plot(val_base.epoch, val_base.loss,label="Baseline")

# plt.plot(train_eca.epoch, train_eca.loss,label="ECA Train")
plt.plot(val_eca.epoch, val_eca.loss,label="ECA")
plt.plot(val_se.epoch, val_se.loss,label="SE")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Val Loss Curve")
plt.legend()

plt.savefig("figures/ablation_loss_curve.png")
plt.close()

# Accuracy Curve
plt.figure()

# plt.plot(train_base.epoch, train_base.acc,label="Baseline Train")
plt.plot(val_base.epoch, val_base.acc,label="Baseline")

# plt.plot(train_eca.epoch, train_eca.acc,label="ECA Train")
plt.plot(val_eca.epoch, val_eca.acc,label="ECA")
plt.plot(val_se.epoch, val_se.acc,label="SE")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Val Accuracy Curve")
plt.legend()

plt.savefig("figures/ablation_accuracy_curve.png")
plt.close()

print("✅ 曲线图已保存到 figures/")