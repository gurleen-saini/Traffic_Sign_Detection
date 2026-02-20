# core/pipeline_rcnn.py

import cv2
import numpy as np

from fast_rcnn.detect_fast_rcnn import detect_frame
from tracker.sort import Sort
from core.distance import estimate_distance
from core.labels import CNN_LABELS, RCNN_ZERO_BASED_LABELS
from tensorflow.keras.models import load_model


class TrafficSignRCNNPipeline:
    def __init__(self):
        # ----------------------------
        # SORT TRACKER (same idea as YOLO tracking)
        # ----------------------------
        self.tracker = Sort(
            max_age=15,
            min_hits=2,
            iou_threshold=0.3
        )

        # ----------------------------
        # CNN CLASSIFIER (same model used with YOLO)
        # ----------------------------
        self.cnn = load_model("model.h5", compile=False)

        # ----------------------------
        # FRAME COUNTER
        # ----------------------------
        self.frame_id = 0

        print("✅ Faster R-CNN pipeline initialized")

    def process_frame(self, frame):
        self.frame_id += 1
        results = []

        # ============================
        # 1️⃣ RCNN DETECTION
        # ============================
        detections = detect_frame(frame, conf=0.5)
        if not detections:
            return results

        dets = []          # SORT input
        det_classes = []   # RCNN class ids aligned with dets

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            class_id = det["class_id"]

            dets.append([x1, y1, x2, y2, score])
            det_classes.append(class_id)

        dets = np.array(dets)

        # ============================
        # 2️⃣ TRACKING (SORT)
        # ============================
        tracks = self.tracker.update(dets)

        # ============================
        # 3️⃣ PROCESS TRACKS
        # ============================
        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)

            # ---- match track to RCNN detection using IOU ----
            best_iou = 0
            best_idx = -1

            for i, d in enumerate(dets):
                dx1, dy1, dx2, dy2 = map(int, d[:4])

                xx1 = max(x1, dx1)
                yy1 = max(y1, dy1)
                xx2 = min(x2, dx2)
                yy2 = min(y2, dy2)

                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_trk = (x2 - x1) * (y2 - y1)
                area_det = (dx2 - dx1) * (dy2 - dy1)
                union = area_trk + area_det - inter

                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx == -1:
                continue

            det_class_id = det_classes[best_idx]
            # detector_label = RCNN_ZERO_BASED_LABELS.get(det_class_id, "unknown")


            # ============================
            # 4️⃣ CNN CLASSIFICATION (FINAL LABEL)
            # ============================
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (32, 32))
            crop = crop.astype("float32") / 255.0
            crop = np.expand_dims(crop, axis=0)

            cnn_pred = self.cnn.predict(crop, verbose=0)
            cnn_class_id = int(np.argmax(cnn_pred))
            cnn_label = CNN_LABELS.get(cnn_class_id, "unknown")

            # ============================
            # 5️⃣ DISTANCE (USE HEIGHT)
            # ============================
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_height)

            # ============================
            # 6️⃣ FINAL RESULT (YOLO-LIKE)
            # ============================
            results.append({
                "track_id": int(track_id),
                # "detector_label": detector_label,   # RCNN (coarse)
                "cnn_label": cnn_label,             # CNN (final)
                "final_label": cnn_label,           # ⭐ match YOLO
                "distance": distance,
                "frame": self.frame_id,
                "model": "faster_rcnn",
                "bbox": [x1, y1, x2, y2]
            })

        return results

# ===== SIMPLE WRAPPER FOR EVALUATION =====
pipeline = TrafficSignRCNNPipeline()

def detect_rcnn(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return []

    results = pipeline.process_frame(frame)

    boxes = []
    for r in results:
        boxes.append(r["bbox"])   # real bbox

    return boxes
