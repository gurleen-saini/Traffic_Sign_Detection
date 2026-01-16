from ultralytics import YOLO

# Load base YOLOv8 model
model = YOLO("yolov8s.pt")

# Train
model.train(
    data="DetectTrafficSignYolov8_data.yaml",
    epochs=100,
    imgsz=640,        # 1024 is heavy for CPU; 640 is safer
    batch=8,
    device="cpu"      # explicit since you're on CPU
)

# Validate
model.val()
