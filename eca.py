import os
# 解决 Windows libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models.mobilenetv2 import InvertedResidual


# ==========================================
# 1. 模型架构定义 (确保与训练时完全一致)
# ==========================================

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
        # 预测时不强制下载权重，后续通过 load_state_dict 加载
        base = models.mobilenet_v2(weights=None)
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


# ==========================================
# 2. Grad-CAM 核心逻辑
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册 Hook 获取特征图和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # 计算通道权重并生成热力图
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        return cam, class_idx


# ==========================================
# 3. 运行、显示与保存逻辑
# ==========================================

def run_grad_cam_and_save(img_path, weight_path, device, save_dir="figures"):
    # 自动创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 标签映射
    labels = ["Early_Blight","Healthy","Late_Blight"]

    # 1. 初始化模型并加载权重
    model = ImprovedMobileNetV2(num_classes=3)
    if not os.path.exists(weight_path):
        print(f"❌ 错误：找不到权重文件 {weight_path}")
        return
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()

    # 2. 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    # 3. 提取热力图
    # MobileNetV2 的 features[18] 是最后的 1x1 卷积层
    target_layer = model.features[18]
    cam_engine = GradCAM(model, target_layer)
    heatmap_raw, class_idx = cam_engine.generate_heatmap(input_tensor)

    # 4. 图像对齐与融合处理
    img_np = np.array(raw_img)
    h, w, _ = img_np.shape
    short_side = min(h, w)
    top = (h - short_side) // 2
    left = (w - short_side) // 2
    img_cropped = img_np[top:top + short_side, left:left + short_side]
    img_cv = cv2.resize(img_cropped, (224, 224))

    heatmap = cv2.resize(heatmap_raw, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加热力图
    overlayed_img = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)

    # 5. 可视化并保存
    plt.figure(figsize=(12, 6), dpi=150)

    plt.subplot(1, 2, 1)
    plt.imshow(img_cv)
    plt.title(f"Original (Pred: {labels[class_idx]})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlayed_img)
    plt.title("Grad-CAM: ECA Attention Focus")
    plt.axis('off')

    # 构造保存文件名 (基于原图片名)
    base_name = os.path.basename(img_path).split('.')[0]
    save_path = os.path.join(save_dir, f"gradcam_{base_name}.png")

    plt.tight_layout()
    plt.savefig(save_path)  # 保存图片
    print(f"✅ 热力图已保存至: {save_path}")

    plt.show()  # 显示图片
    plt.close()


# ==========================================
# 主程序入口
# ==========================================

if __name__ == '__main__':
    # --- 配置参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_WEIGHTS = "models/eca_best.pth"

    # 请替换为你想要分析的图片路径
    TEST_IMAGE = r'E:\Source\potato\dataset\Testing\Early_Blight\Early_Blight_34.jpg'

    if os.path.exists(TEST_IMAGE):
        run_grad_cam_and_save(TEST_IMAGE, MODEL_WEIGHTS, DEVICE)
    else:
        print(f"⚠️ 找不到测试图片: {TEST_IMAGE}")