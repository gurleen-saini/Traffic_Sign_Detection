import cv2
import torch
import numpy as np

# Load YOLOv7 model
device = "cpu"
model = torch.hub.load(
    "WongKinYiu/yolov7",
    "custom",
    path="yolov7/runs/train/yolov7_traffic_sign/weights/best.pt",
    source="github",
    force_reload=False
)
model.to(device)
model.eval()

video = cv2.VideoCapture("traffic-sign-to-test.mp4")

PRINT_INTERVAL = 0.5
last_print = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    for *box, conf, cls in detections:
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)

        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - last_print > PRINT_INTERVAL:
            print(f"[YOLOv7] Class={cls} Conf={conf:.2f}")
            last_print = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"YOLOv7 | C:{cls} | {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow("YOLOv7 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
