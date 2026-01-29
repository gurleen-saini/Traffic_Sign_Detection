from tensorflow.keras.models import load_model
import cv2
import numpy as np

cnn_model = load_model("model.h5", compile=False)

def classify(crop):
    crop = cv2.resize(crop, (32, 32))
    crop = np.expand_dims(crop, axis=0)
    pred = cnn_model.predict(crop, verbose=0)
    return int(pred.argmax())
