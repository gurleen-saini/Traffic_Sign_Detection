import os
import cv2
import torch
from torch.utils.data import Dataset


class YOLOtoSSDDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print("===================================")
        print("[SSD DATASET INIT]")
        print("Image dir :", img_dir)
        print("Label dir :", label_dir)
        print("Total images found:", len(self.images))
        print("===================================")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label_path = os.path.join(
            self.label_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())

                    if bw <= 0 or bh <= 0:
                        print(f"[SKIP] Invalid bbox in {label_path}")
                        continue

                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h

                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)
        else:
            print(f"[WARN] Missing label file: {label_path}")

        if len(boxes) == 0:
            print(f"[SKIP] No boxes for image: {img_name}")
            return None

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return img, target
