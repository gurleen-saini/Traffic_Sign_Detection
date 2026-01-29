import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "model.h5"
TEST_DIR = "cnn_test_dataset_fixed"  # class-wise folders 1–43

model = load_model(MODEL_PATH, compile=False)

y_true = []
y_pred = []

for cls in os.listdir(TEST_DIR):
    cls_path = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    print(f"\n🔍 Processing class {cls} ...")

    count = 0
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (32, 32))
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        y_pred.append(np.argmax(pred) + 1)
        y_true.append(int(cls))

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} images")

    print(f"✅ Finished class {cls} ({count} images)")


# Accuracy
acc = accuracy_score(y_true, y_pred)
print("CNN Accuracy:", acc)

# Detailed report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix Shape:", cm.shape)
