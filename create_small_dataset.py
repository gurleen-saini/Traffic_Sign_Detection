import os
import shutil
import random

SOURCE_DIR = "cnn_test_dataset_fixed"
DEST_DIR = "cnn_test_small"

IMAGES_PER_CLASS = 100

os.makedirs(DEST_DIR, exist_ok=True)

for cls in os.listdir(SOURCE_DIR):

    src_cls_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(src_cls_path):
        continue

    dst_cls_path = os.path.join(DEST_DIR, cls)

    os.makedirs(dst_cls_path, exist_ok=True)

    images = os.listdir(src_cls_path)

    random.shuffle(images)

    selected_images = images[:IMAGES_PER_CLASS]

    for img_name in selected_images:

        src_img = os.path.join(src_cls_path, img_name)

        dst_img = os.path.join(dst_cls_path, img_name)

        shutil.copy(src_img, dst_img)

    print(f"Copied {len(selected_images)} images for class {cls}")

print("\n✅ Small balanced dataset created successfully!")