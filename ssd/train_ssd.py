import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from dataset_yolo_to_ssd import YOLOtoSSDDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = YOLOtoSSDDataset(
    img_dir="../train/images",
    label_dir="../train/labels"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

model = ssd300_vgg16(pretrained=True)
model.head.classification_head.num_classes = 44  # 43 signs + background
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(10):
    epoch_loss = 0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[SSD] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "ssd_traffic_sign.pt")
