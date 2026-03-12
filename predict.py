import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['Early_Blight', 'Healthy', 'Late_Blight']  # 这里的顺序必须和文件夹字母顺序一致

# 2. 加载模型架构 (必须和训练时完全一致)
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 3)

# 3. 加载训练好的权重
model.load_state_dict(torch.load('potato_mobilenetv3_baseline.pth'))
model = model.to(device)
model.eval()


# 4. 预处理函数
def predict_image(image_path):
    # 按照 MobileNetV3 论文要求的尺寸和归一化处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # 增加 Batch 维度

    with torch.no_grad():
        outputs = model(img_tensor)
        # 论文知识点：使用 Softmax 将输出转化为概率
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    class_name = classes[predicted.item()]
    print(f"预测结果: {class_name} ({confidence.item() * 100:.2f}%)")


# 5. 执行预测 (请替换成你的图片文件名)
if __name__ == "__main__":
    # 示例：预测 Testing 文件夹下的一张图
    test_img = r"E:\Source\potato\dataset\Testing\Healthy\Healthy_20.jpg"
    if os.path.exists(test_img):
        predict_image(test_img)
    else:
        print("未找到测试图片，请确认路径。")