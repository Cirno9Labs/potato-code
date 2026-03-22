import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

baseline = pd.read_csv("logs/baseline_log.csv")
eca = pd.read_csv("logs/eca_log.csv")

train_base = baseline[baseline.phase=="train"]
val_base = baseline[baseline.phase=="val"]

train_eca = eca[eca.phase=="train"]
val_eca = eca[eca.phase=="val"]

# Loss Curve
plt.figure()

plt.plot(train_base.epoch, train_base.loss,label="Baseline Train")
plt.plot(val_base.epoch, val_base.loss,label="Baseline Val")

plt.plot(train_eca.epoch, train_eca.loss,label="ECA Train")
plt.plot(val_eca.epoch, val_eca.loss,label="ECA Val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.savefig("figures/loss_curve.png")
plt.close()

# Accuracy Curve
plt.figure()

plt.plot(train_base.epoch, train_base.acc,label="Baseline Train")
plt.plot(val_base.epoch, val_base.acc,label="Baseline Val")

plt.plot(train_eca.epoch, train_eca.acc,label="ECA Train")
plt.plot(val_eca.epoch, val_eca.acc,label="ECA Val")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()

plt.savefig("figures/accuracy_curve.png")
plt.close()

print("✅ 曲线图已保存到 figures/")