import json
from collections import Counter
from sklearn.metrics import accuracy_score

# ================= LOAD FILES =================
with open("pipeline_output.json", "r") as f:
    predictions = json.load(f)["results"]

with open("ground_truth.json", "r") as f:
    gt_objects = json.load(f)["objects"]

# ================= RULE MAPPING =================
def map_to_rule(label):
    if "Speed limit" in label:
        return "SPEED_LIMIT"
    if "No passing" in label:
        return "NO_PASSING"
    return "OTHER"

# ================= GROUND TRUTH MAP =================
gt_map = {}
for obj in gt_objects:
    gt_map[obj["track_id"]] = map_to_rule(obj["true_label"])

# ================= COLLECT PREDICTIONS PER TRACK =================
track_predictions = {}

for pred in predictions:
    track_id = pred["track_id"]
    rule_label = map_to_rule(pred["cnn_label"])

    track_predictions.setdefault(track_id, []).append(rule_label)

# ================= FINAL RULE PER TRACK (MAJORITY VOTE) =================
y_true = []
y_pred = []

print("\n===== RULE-LEVEL DECISIONS =====")
for track_id, preds in track_predictions.items():
    final_pred = Counter(preds).most_common(1)[0][0]
    true_rule = gt_map.get(track_id)

    y_true.append(true_rule)
    y_pred.append(final_pred)

    print(f"Track {track_id}: TRUE={true_rule} | PRED={final_pred}")

# ================= ACCURACY =================
accuracy = accuracy_score(y_true, y_pred)

print("\n===== RULE-LEVEL ACCURACY =====")
print(f"Accuracy: {accuracy:.2f}")
