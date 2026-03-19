import cv2
import numpy as np
import os
import glob
import random
import shutil

# --- 配置部分 ---
# 请将此路径修改为你自己的 PLD 原始数据集路径
ORIGINAL_TRAIN_DIR = r'E:\Source\potato\dataset\Training'
# 增强后数据的保存路径
AUGMENTED_TRAIN_DIR = r"E:\Source\potato\dataset\Training_Augmented"

# 定义类别
CLASSES = ['Early_Blight', 'Late_Blight', 'Healthy']

# 定义增强参数（保持不变）
BRIGHTNESS_RANGE = (0.7, 1.3)
GAUSSIAN_MU = 0
GAUSSIAN_SIGMA = 25

# --- 新增：按操作名称创建的子文件夹名称 ---
OPERATION_DIRS = {
    'original': '原图',
    'horizontal': '水平镜像 + 随机亮度调整',
    'vertical':  '垂直翻转 + 随机亮度调整',
    'noise':     '添加高斯噪声'
}

# --- 辅助函数（保持不变）---
def adjust_brightness(image, range=(0.7, 1.3)):
    ratio = random.uniform(range[0], range[1])
    brightened = np.clip(image.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
    return brightened

def add_gaussian_noise(image, mu=0, sigma=25):
    h, w, c = image.shape
    noise = np.random.normal(mu, sigma, (h, w, c))
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

# --- 主程序 ---
def run_augmentation():
    print("开始数据增强程序...")

    # 1. 创建目标根目录
    if not os.path.exists(AUGMENTED_TRAIN_DIR):
        os.makedirs(AUGMENTED_TRAIN_DIR)
        print(f"已创建增强后文件夹: {AUGMENTED_TRAIN_DIR}")
    else:
        print(f"警告: 目标文件夹已存在，可能会覆盖旧数据！")

    total_original = 0
    total_augmented = 0

    # 2. 遍历每个类别
    for cls in CLASSES:
        print(f"正在处理类别: {cls}...")

        # 创建类别文件夹
        cls_aug_dir = os.path.join(AUGMENTED_TRAIN_DIR, cls)
        if not os.path.exists(cls_aug_dir):
            os.makedirs(cls_aug_dir)

        # === 新增：为该类别创建 4 个操作子文件夹 ===
        for folder_name in OPERATION_DIRS.values():
            sub_dir = os.path.join(cls_aug_dir, folder_name)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # 查找原始图片
        cls_orig_dir = os.path.join(ORIGINAL_TRAIN_DIR, cls)
        image_paths = glob.glob(os.path.join(cls_orig_dir, "*.jpg"))

        print(f" > 找到原始图片数量: {len(image_paths)}")
        total_original += len(image_paths)

        # 3. 遍历每张图片
        for i, img_path in enumerate(image_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                continue

            base_name = f"{img_name}.jpg"   # 所有文件夹内都使用和原图一模一样的文件名

            # ── 原图 ──
            orig_dir = os.path.join(cls_aug_dir, OPERATION_DIRS['original'])
            shutil.copy(img_path, os.path.join(orig_dir, base_name))

            # ── 水平镜像 + 随机亮度调整 ──
            img_b = cv2.flip(img, 1)
            img_b_aug = adjust_brightness(img_b, range=BRIGHTNESS_RANGE)
            horiz_dir = os.path.join(cls_aug_dir, OPERATION_DIRS['horizontal'])
            cv2.imwrite(os.path.join(horiz_dir, base_name), img_b_aug)
            total_augmented += 1

            # ── 垂直翻转 + 随机亮度调整 ──
            img_c = cv2.flip(img, 0)
            img_c_aug = adjust_brightness(img_c, range=BRIGHTNESS_RANGE)
            vert_dir = os.path.join(cls_aug_dir, OPERATION_DIRS['vertical'])
            cv2.imwrite(os.path.join(vert_dir, base_name), img_c_aug)
            total_augmented += 1

            # ── 添加高斯噪声 ──
            img_d_aug = add_gaussian_noise(img, mu=GAUSSIAN_MU, sigma=GAUSSIAN_SIGMA)
            noise_dir = os.path.join(cls_aug_dir, OPERATION_DIRS['noise'])
            cv2.imwrite(os.path.join(noise_dir, base_name), img_d_aug)
            total_augmented += 1

            if (i + 1) % 100 == 0:
                print(f" 已完成类别 {cls} 的 {i+1} 张图片的处理.")

    # 4. 最终统计
    print("-" * 30)
    print("数据增强完成！")
    print(f"原始训练图片总数: {total_original}")
    print(f"生成的新图片数量: {total_augmented}")
    print(f"最终训练集总数（原始+增强）: {total_original + total_augmented}")
    print(f"增强后的数据结构如下：")
    print(f"   {AUGMENTED_TRAIN_DIR}")
    print(f"   ├── Early_Blight/")
    print(f"   │   ├── 原图/")
    print(f"   │   ├── 水平镜像 + 随机亮度调整/")
    print(f"   │   ├── 垂直翻转 + 随机亮度调整/")
    print(f"   │   └── 添加高斯噪声/")
    print(f"   └── ...（其他类别同上）")
    print(f"每个子文件夹内图片文件名与原图完全一致！")

if __name__ == "__main__":
    run_augmentation()