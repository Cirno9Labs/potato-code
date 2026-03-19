import os
import random
import shutil
from glob import glob

# --- 配置部分 ---
ORIGINAL_TRAIN_DIR = r"E:\Source\potato\dataset\Training_Augmented"
AUGMENTED_TRAIN_DIR = r"E:\Source\potato\dataset\Training_Augmented_Selected"

# 设定目标总数。3264 可以达成各类别完美平衡
TARGET_COUNT = 3264

CLASSES = ['Early_Blight', 'Late_Blight', 'Healthy']

DIR_ORIGINAL = '原图'
AUG_DIRS = [
    '水平镜像 + 随机亮度调整',
    '垂直翻转 + 随机亮度调整',
    '添加高斯噪声'
]


def balanced_selection_nested():
    if not os.path.exists(AUGMENTED_TRAIN_DIR):
        os.makedirs(AUGMENTED_TRAIN_DIR)
        print(f"已创建目标文件夹: {AUGMENTED_TRAIN_DIR}")

    for cls in CLASSES:
        print(f"\n====================================")
        print(f"正在处理类别: {cls}")

        src_cls_path = os.path.join(ORIGINAL_TRAIN_DIR, cls)
        dst_cls_path = os.path.join(AUGMENTED_TRAIN_DIR, cls)

        # 预先在目标路径创建原图和所有增强操作的子文件夹
        os.makedirs(os.path.join(dst_cls_path, DIR_ORIGINAL), exist_ok=True)
        for aug_dir in AUG_DIRS:
            os.makedirs(os.path.join(dst_cls_path, aug_dir), exist_ok=True)

        # 1. 收集并处理原图 (全量保留)
        orig_dir_path = os.path.join(src_cls_path, DIR_ORIGINAL)
        origin_images = glob(os.path.join(orig_dir_path, "*.jpg"))
        print(f"  > 找到原图: {len(origin_images)} 张，全部保留。")

        for img_path in origin_images:
            # 保持原文件名，放入目标的“原图”子文件夹
            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(dst_cls_path, DIR_ORIGINAL, base_name))

        # 2. 计算需要补充的数量
        needed = TARGET_COUNT - len(origin_images)
        if needed <= 0:
            print("  > 原图已达到或超过目标数量，无需从增强池补充。")
            continue

        # 3. 收集增强图池
        # 记录格式: [(文件绝对路径, 所属的增强子文件夹名称), ...]
        aug_images_pool = []
        for aug_dir in AUG_DIRS:
            aug_dir_path = os.path.join(src_cls_path, aug_dir)
            images_in_dir = glob(os.path.join(aug_dir_path, "*.jpg"))
            for img_path in images_in_dir:
                aug_images_pool.append((img_path, aug_dir))

        print(f"  > 找到增强图池: {len(aug_images_pool)} 张，需要随机抽取: {needed} 张。")

        # 4. 随机抽样
        selected_augs = []
        if len(aug_images_pool) >= needed:
            selected_augs = random.sample(aug_images_pool, needed)
        else:
            print(f"  [警告] {cls} 类别增强图库存不足，已取走全部图片！")
            selected_augs = aug_images_pool

        # 5. 执行拷贝 (放回对应的子目录，保持原名)
        for img_path, aug_dir in selected_augs:
            base_name = os.path.basename(img_path)
            # 目标路径: 根目录/类别/具体增强操作文件夹/原文件名.jpg
            dst_path = os.path.join(dst_cls_path, aug_dir, base_name)
            shutil.copy(img_path, dst_path)

        # 6. 统计该类别最终写入总数量 (遍历该类别下所有子文件夹的jpg)
        final_count = len(glob(os.path.join(dst_cls_path, "*", "*.jpg")))
        print(f"  [完成] {cls} 最终写入总数量: {final_count} 张")

    print(f"\n====================================")
    print("所有类别的嵌套结构平衡抽取任务已完成！")


if __name__ == "__main__":
    balanced_selection_nested()