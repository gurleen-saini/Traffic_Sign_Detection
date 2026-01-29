# from ultralytics import YOLO

# # Load base YOLOv8 model
# model = YOLO("yolov8s.pt")

# # Train
# model.train(
#     data="DetectTrafficSignYolov8_data.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=8,
#     device="cpu",
#     project="runs/detect",
#     name="train",
#     resume=True
# )

# # Validate
# model.val()

from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/last.pt")

model.train(
    data="DetectTrafficSignYolov8_data.yaml",
    epochs=100,          
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs/detect",
    name="train",
    resume=True          
)

model.val()