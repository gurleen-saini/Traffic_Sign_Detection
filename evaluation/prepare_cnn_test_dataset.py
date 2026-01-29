import os
import shutil

# SOURCE: your current dataset (0–42)
SRC_DIR = r"C:\Users\Windows\Desktop\DetectTrafficSign\cnn_test_dataset"

# DESTINATION: corrected dataset (1–43)
DST_DIR = r"C:\Users\Windows\Desktop\DetectTrafficSign\cnn_test_dataset_fixed"

os.makedirs(DST_DIR, exist_ok=True)

for folder in os.listdir(SRC_DIR):
    src_class_path = os.path.join(SRC_DIR, folder)

    if not os.path.isdir(src_class_path):
        continue

    old_class = int(folder)      # 0–42
    new_class = old_class + 1    # 1–43

    dst_class_path = os.path.join(DST_DIR, str(new_class))
    os.makedirs(dst_class_path, exist_ok=True)

    for img in os.listdir(src_class_path):
        src_img = os.path.join(src_class_path, img)
        dst_img = os.path.join(dst_class_path, img)

        if os.path.isfile(src_img):
            shutil.copy(src_img, dst_img)

    print(f"Class {old_class} → {new_class} copied")

print("\n✅ CNN test dataset prepared successfully!")
