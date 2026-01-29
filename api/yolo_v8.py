from ultralytics import YOLO

yolo_model = YOLO(r"runs\detect\train\weights\best.pt")

def detect(frame):
    results = yolo_model(frame, conf=0.4)
    return results
