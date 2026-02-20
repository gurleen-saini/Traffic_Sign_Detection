import cv2
import torch
from pathlib import Path
from torchvision.models.detection import ssd300_vgg16

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CONFIG =================
NUM_CLASSES = 44   # background + 43 traffic signs

# ================= LOAD SSD MODEL =================
# weights=None → no COCO weights
# weights_backbone=None → stops VGG internet download
model = ssd300_vgg16(
    weights=None,
    weights_backbone=None
)

# Change classifier head to your dataset classes
model.head.classification_head.num_classes = NUM_CLASSES

# ================= LOAD TRAINED WEIGHTS =================
# Go ONE folder up because .pt is in root DetectTrafficSign/
WEIGHTS_PATH = Path(__file__).parent.parent / "ssd_traffic_sign.pt"

print(f"📦 Loading SSD weights from: {WEIGHTS_PATH}")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ SSD model loaded successfully")

# ================= DETECTION FUNCTION =================
def detect_frame(frame, conf=0.4):
    if frame is None:
        return []

    # Convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize + Tensor
    img_tensor = (
        torch.from_numpy(img / 255.0)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        preds = model(img_tensor)[0]

    results = []

    for box, score, label in zip(
        preds["boxes"],
        preds["scores"],
        preds["labels"]
    ):
        if score < conf:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        results.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(label)   # coarse SSD class
        })

    return results
