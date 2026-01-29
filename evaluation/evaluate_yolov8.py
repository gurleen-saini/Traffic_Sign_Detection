from ultralytics import YOLO

model = YOLO(r"runs\detect\train\weights\best.pt")

metrics = model.val(
    data="DetectTrafficSignYolov8_data.yaml",
    imgsz=640,
    batch=8,
    conf=0.25
)

print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

# import os

# # ====== CONFIG ======
# SPLITS = ["train", "valid"]
# OLD_LABEL_DIR_NAME = "labels"        # original labels
# NEW_LABEL_DIR_NAME = "labels_4cls"   # new YOLO labels

# # ====== CLASS MAPPING ======
# PROHIBITORY = [0,1,2,3,4,5,7,8,9,10,15,16,17]
# DANGER = [18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# MANDATORY = [33,34,35,36,37,38,39,40]
# OTHER = [6,11,12,13,14,32,41,42]

# def map_class(cls_id):
#     if cls_id in PROHIBITORY:
#         return 0
#     elif cls_id in DANGER:
#         return 1
#     elif cls_id in MANDATORY:
#         return 2
#     elif cls_id in OTHER:
#         return 3
#     else:
#         return None   # ignore unknown

# # ====== PROCESS ======
# for split in SPLITS:
#     old_label_dir = os.path.join(split, OLD_LABEL_DIR_NAME)
#     new_label_dir = os.path.join(split, NEW_LABEL_DIR_NAME)

#     os.makedirs(new_label_dir, exist_ok=True)

#     for file in os.listdir(old_label_dir):
#         if not file.endswith(".txt"):
#             continue

#         old_path = os.path.join(old_label_dir, file)
#         new_path = os.path.join(new_label_dir, file)

#         new_lines = []

#         with open(old_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 old_cls = int(parts[0])
#                 new_cls = map_class(old_cls)

#                 if new_cls is None:
#                     continue

#                 parts[0] = str(new_cls)
#                 new_lines.append(" ".join(parts))

#         with open(new_path, "w") as f:
#             f.write("\n".join(new_lines))

#     print(f"✅ {split} labels converted successfully")

# print("🎯 YOLOv8 dataset is now correctly mapped to 4 classes")
