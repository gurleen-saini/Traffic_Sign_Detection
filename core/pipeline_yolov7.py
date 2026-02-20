import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tracker.sort import Sort
import torch.serialization
import numpy as np

# Allow YOLOv7 old weights to load in PyTorch 2.6+
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray
])


from core.slam import VisualSLAM
from core.distance import estimate_distance
from core.labels import CLASSES_43


class TrafficSignYOLOv7Pipeline:
    def __init__(self):
        self.yolo = torch.hub.load(
            'yolov7',   # <-- LOCAL FOLDER NAME
            'custom',
            'yolov7/runs/train/ts_4class8/weights/best.pt',
            source='local'
        )


        self.cnn = load_model("model.h5", compile=False)
        self.tracker = Sort()
        self.slam = VisualSLAM()

    def detect_with_yolo(self, frame):
        detections, crops, labels, boxes = [], [], [], []

        results = self.yolo(frame)
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = float(row['confidence'])

            if conf < 0.4:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            detections.append([x1, y1, x2, y2, conf])
            crops.append(crop)
            labels.append(row['name'])
            boxes.append([x1, y1, x2, y2])

        return detections, crops, labels, boxes

    def process_frame(self, frame):
        detections, crops, labels, boxes = self.detect_with_yolo(frame)

        results = []

        for box in boxes:
            x1, y1, x2, y2 = box
            results.append({
                "bbox": [x1, y1, x2, y2]
            })

        return results

# ===== SIMPLE WRAPPER FOR EVALUATION =====
pipeline = TrafficSignYOLOv7Pipeline()

def detect_yolov7(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return []

    results = pipeline.process_frame(frame)

    boxes = []
    for r in results:
        boxes.append(r["bbox"])

    return boxes
