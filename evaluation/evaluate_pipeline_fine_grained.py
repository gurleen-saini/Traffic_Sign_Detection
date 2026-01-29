import json
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# ================= LOAD FILES =================
with open("pipeline_output.json", "r") as f:
    predictions = json.load(f)["results"]

with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)["objects"]

# ================= BUILD GT MAP (TRACK-WISE) =================
gt_map = {obj["track_id"]: obj["true_label"] for obj in ground_truth}

y_true = []
y_pred = []

# ================= ALIGN PREDICTIONS =================
for pred in predictions:
    track_id = pred["track_id"]
    if track_id in gt_map:
        y_true.append(gt_map[track_id])
        y_pred.append(pred["cnn_label"])

# ================= CONFUSION MATRIX =================
labels = sorted(list(set(y_true + y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

print("\n===== PIPELINE CONFUSION MATRIX =====")
print(df_cm)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, zero_division=0))


# ================= TRACK STABILITY METRIC =================
track_history = defaultdict(list)

for pred in predictions:
    track_history[pred["track_id"]].append(pred["cnn_label"])

stable_tracks = 0
for track_id, label_seq in track_history.items():
    if len(set(label_seq)) == 1:
        stable_tracks += 1

stability_score = stable_tracks / len(track_history)

print("\n===== TRACK STABILITY =====")
print(f"Stable Tracks: {stable_tracks}/{len(track_history)}")
print(f"Stability Score: {stability_score:.2f}")
