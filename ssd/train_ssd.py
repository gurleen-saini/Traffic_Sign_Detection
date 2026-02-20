import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

from dataset_yolo_to_ssd import YOLOtoSSDDataset

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===================================")
print("Using device:", device)
print("===================================")

# ---------------- DATASET ----------------
dataset = YOLOtoSSDDataset(
    img_dir="train/images",
    label_dir="train/labels"
)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

print("[INFO] DataLoader created")

# ---------------- MODEL ----------------
NUM_CLASSES = 44

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model.head.classification_head.num_classes = NUM_CLASSES
model.to(device)

print("[INFO] SSD model initialized")

# ---------------- OPTIMIZER ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- TRAINING ----------------
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    used_batches = 0

    print(f"\n========== EPOCH {epoch+1}/{NUM_EPOCHS} ==========")

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            print(f"[E{epoch+1}] Batch {batch_idx} skipped (empty)")
            continue

        images, targets = batch

        print(
            f"[E{epoch+1}] Batch {batch_idx} | "
            f"Images: {len(images)} | "
            f"Boxes per image: {[len(t['boxes']) for t in targets]}"
        )

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        used_batches += 1

        print(
            f"[E{epoch+1}] Batch {batch_idx} Loss: {loss.item():.4f}"
        )

    if used_batches == 0:
        print("[ERROR] No valid batches used in this epoch!")
    else:
        print(
            f"[SSD] Epoch {epoch+1} DONE | "
            f"Avg Loss: {epoch_loss / used_batches:.4f} | "
            f"Batches used: {used_batches}"
        )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "ssd_traffic_sign.pt")
print("===================================")
print("✅ SSD model saved as ssd_traffic_sign.pt")
print("===================================")
