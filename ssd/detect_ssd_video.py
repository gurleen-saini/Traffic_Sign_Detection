import cv2
import torch
import numpy as np
from torchvision.models.detection import ssd300_vgg16
from tensorflow.keras.models import load_model

device = torch.device("cpu")

cnn_model = load_model("../model.h5", compile=False)

model = ssd300_vgg16(pretrained=False)
model.head.classification_head.num_classes = 44
model.load_state_dict(torch.load("ssd_traffic_sign.pt", map_location=device))
model.eval().to(device)

video = cv2.VideoCapture("../traffic-sign-to-test.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img / 255.0).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    for box, score, label in zip(
        outputs["boxes"], outputs["scores"], outputs["labels"]
    ):
        if score < 0.4:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (32, 32))
        pred = cnn_model.predict(np.expand_dims(crop, 0), verbose=0)
        cnn_class = np.argmax(pred) + 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"SSD | CNN:{cnn_class} | {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow("SSD + CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
