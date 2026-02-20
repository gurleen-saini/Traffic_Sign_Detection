import cv2
import numpy as np

from ssd.detect_ssd import detect_frame
from tracker.sort import Sort
from core.distance import estimate_distance
from core.labels import CNN_LABELS
from tensorflow.keras.models import load_model


class TrafficSignSSDPipeline:
    def __init__(self):
        self.tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)
        self.cnn = load_model("model.h5", compile=False)
        self.frame_id = 0

        print("✅ SSD pipeline initialized")

    def process_frame(self, frame):
        self.frame_id += 1
        results = []

        detections = detect_frame(frame, conf=0.4)
        if not detections:
            return results

        dets = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            dets.append([x1, y1, x2, y2, d["score"]])

        dets = np.array(dets)
        tracks = self.tracker.update(dets)

        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (32, 32))
            crop = crop.astype("float32") / 255.0
            crop = np.expand_dims(crop, axis=0)

            pred = self.cnn.predict(crop, verbose=0)
            cnn_class_id = int(np.argmax(pred))
            cnn_label = CNN_LABELS.get(cnn_class_id, "unknown")

            distance = estimate_distance(y2 - y1)

            results.append({
                "track_id": int(track_id),
                "cnn_label": cnn_label,
                "final_label": cnn_label,
                "distance": distance,
                "frame": self.frame_id,
                "model": "ssd",
                "bbox": [x1, y1, x2, y2] 
            })

        return results
    
# ===== SIMPLE WRAPPER FOR EVALUATION =====
# pipeline = TrafficSignSSDPipeline()

# def detect_ssd(image_path):
#     frame = cv2.imread(image_path)
#     if frame is None:
#         return []

#     results = pipeline.process_frame(frame)

#     boxes = []
#     for r in results:
#         boxes.append(r["bbox"])   # REAL BBOX

#     return boxes
