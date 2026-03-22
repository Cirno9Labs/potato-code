import os
import shutil
import glob

# --- 配置路径 ---
DATASET_DIR = r"E:\Source\potato\dataset"
# 原始训练集路径（用于获取基础数量）
ORIGINAL_TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
# 抽样后的增强数据路径（源数据所在）
AUG_SELECTED_DIR = os.path.join(DATASET_DIR, "Training_Augmented_Selected")
# 最终整合的输出路径
TARGET_TRAIN_DIR = os.path.join(AUG_SELECTED_DIR, "Training")

# 定义类别
CLASSES = ['Early_Blight', 'Healthy', 'Late_Blight']

# 需要从中提取图片的增强子文件夹（不包含“原图”）
AUG_FOLDERS = [
    '垂直翻转 + 随机亮度调整',
    '水平镜像 + 随机亮度调整',
    '添加高斯噪声'
]


def rename_and_flatten_augmented_data():
    print("开始执行重命名与整合任务...\n")

    # 创建最外层的目标 Training 文件夹
    if not os.path.exists(TARGET_TRAIN_DIR):
        os.makedirs(TARGET_TRAIN_DIR)

    for cls in CLASSES:
        print(f"====================================")
        print(f"正在处理类别: {cls}")

        # 1. 统计原始 Training 文件夹下的图片数量，确定起始索引
        orig_cls_dir = os.path.join(ORIGINAL_TRAIN_DIR, cls)
        orig_images = glob.glob(os.path.join(orig_cls_dir, "*.jpg"))

        # 你的例子：如果有1303个文件，从1304开始
        start_index = len(orig_images) + 1
        current_index = start_index

        print(f"  > 原始训练集数量: {len(orig_images)} 张")
        print(f"  > 新增强图片将从 {cls}_{start_index}.jpg 开始命名")

        # 2. 创建该类别的目标子文件夹
        target_cls_dir = os.path.join(TARGET_TRAIN_DIR, cls)
        os.makedirs(target_cls_dir, exist_ok=True)

        # 3. 遍历指定的 3 个增强子文件夹，提取并重命名图片
        copied_count = 0
        for aug_folder in AUG_FOLDERS:
            aug_folder_path = os.path.join(AUG_SELECTED_DIR, cls, aug_folder)

            # 如果该文件夹存在，则读取里面的 jpg 文件
            if os.path.exists(aug_folder_path):
                aug_images = glob.glob(os.path.join(aug_folder_path, "*.jpg"))

                for img_path in aug_images:
                    # 构造新的文件名，例如：Early_Blight_1304.jpg
                    new_filename = f"{cls}_{current_index}.jpg"
                    target_img_path = os.path.join(target_cls_dir, new_filename)

                    # 复制文件并重命名 (使用 copy 避免破坏你的源文件)
                    shutil.copy(img_path, target_img_path)

                    current_index += 1
                    copied_count += 1
            else:
                print(f"  [提示] 未找到路径: {aug_folder_path}")

        print(f"  > [完成] 共提取并重命名了 {copied_count} 张增强图片")
        print(f"  > 最终编号已到达: {cls}_{current_index - 1}.jpg")

    print(f"\n====================================")
    print("所有类别的处理任务已圆满完成！")
    print(f"你可以在此路径下查看结果: {TARGET_TRAIN_DIR}")


if __name__ == "__main__":
    rename_and_flatten_augmented_data()