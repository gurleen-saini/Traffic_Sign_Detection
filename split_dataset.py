import os
import shutil
import random

# ----------------------------
# PATHS
# ----------------------------
SOURCE_DIR = "ts"
TRAIN_IMG_DIR = "train/images"
TRAIN_LBL_DIR = "train/labels"
VAL_IMG_DIR   = "valid/images"
VAL_LBL_DIR   = "valid/labels"

SPLIT_RATIO = 0.8  # 80% train, 20% val

# ----------------------------
# CREATE DIRECTORIES
# ----------------------------
for path in [
    TRAIN_IMG_DIR, TRAIN_LBL_DIR,
    VAL_IMG_DIR, VAL_LBL_DIR
]:
    os.makedirs(path, exist_ok=True)

# ----------------------------
# COLLECT IMAGES
# ----------------------------
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg")]
random.shuffle(images)

split_index = int(len(images) * SPLIT_RATIO)
train_images = images[:split_index]
val_images = images[split_index:]

# ----------------------------
# MOVE FILES
# ----------------------------
def move_files(image_list, img_dest, lbl_dest):
    for img in image_list:
        label = img.replace(".jpg", ".txt")

        shutil.copy(
            os.path.join(SOURCE_DIR, img),
            os.path.join(img_dest, img)
        )

        shutil.copy(
            os.path.join(SOURCE_DIR, label),
            os.path.join(lbl_dest, label)
        )

move_files(train_images, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
move_files(val_images, VAL_IMG_DIR, VAL_LBL_DIR)

print("✅ Dataset split completed")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
