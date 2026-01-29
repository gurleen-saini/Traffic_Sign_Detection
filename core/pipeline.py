# core/pipeline.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tracker.sort import Sort

from core.slam import VisualSLAM
from core.distance import estimate_distance
from core.labels import CLASSES_43


class TrafficSignPipeline:
    def __init__(self):
        self.yolo = YOLO("bestDetectTrafficSign.pt")
        self.cnn = load_model("model.h5", compile=False)
        self.tracker = Sort()
        self.slam = VisualSLAM()

        self.alerted_ids = set()
        self.last_print_time = {}
        self.PRINT_INTERVAL = 0.5

    def detect_with_yolo(self, frame):
        detections, crops, labels, boxes = [], [], [], []

        results = self.yolo.predict(frame, conf=0.5, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)

                if conf < 0.4:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                detections.append([x1, y1, x2, y2, conf])
                crops.append(crop)
                labels.append(f"{self.yolo.model.names[cls_id]} ({conf:.2f})")
                boxes.append([x1, y1, x2, y2])

        return detections, crops, labels, boxes

    def process_frame(self, frame):
        results_out = []

        slam_state = self.slam.update(frame)
        if slam_state != "TRACKING":
            return results_out

        detections, crops, yolo_labels, yolo_boxes = self.detect_with_yolo(frame)
        detections = np.array(detections) if len(detections) else np.empty((0, 5))
        tracks = self.tracker.update(detections)

        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)

            # match YOLO box
            best_iou, best_idx = 0, -1
            for i, box in enumerate(yolo_boxes):
                inter = self._iou([x1,y1,x2,y2], box)
                if inter > best_iou:
                    best_iou, best_idx = inter, i

            if best_idx == -1:
                continue

            crop = crops[best_idx]
            crop = cv2.resize(crop, (32, 32))
            crop = np.expand_dims(crop, axis=0)

            pred = self.cnn.predict(crop, verbose=0)
            cls_id = np.argmax(pred) + 1
            cnn_label = CLASSES_43.get(cls_id, "Unknown")

            distance = estimate_distance(yolo_boxes[best_idx][3] - yolo_boxes[best_idx][1])

            results_out.append({
                "track_id": int(track_id),
                "yolo_label": yolo_labels[best_idx],
                "cnn_label": cnn_label,
                "distance": distance
            })

        return results_out

    def _iou(self, a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)
