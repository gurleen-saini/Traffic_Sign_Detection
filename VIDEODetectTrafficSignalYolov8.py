# -*- coding: utf-8 -*-
"""
Created on Jan 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
#
# Downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data

dirVideo ="traffic-sign-to-test.mp4"
# dirVideo ="4434242-uhd_2160_3840_24fps.mp4"
# dirVideo ="testvideo1.mp4"

#Downloaded from https://www.pexels.com/video/road-trip-4434242/
#dirVideo ="production_id_4434242 (2160p).mp4"

#dirnameYolo="runs\\detect\\train2\\weights\\best.pt"
dirnameYolo="bestDetectTrafficSign.pt"

import cv2
import time

from tracker.sort import Sort
import numpy as np

from slam.slam_runner import SlamSystem
slam = SlamSystem()


tracker = Sort()
alerted_ids = set()

TimeIni=time.time()
# in  14 minutes = 800 seconds finish  
TimeLimit=1000


# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
print(class_list)

import numpy as np


###########################################################
# MAIN
##########################################################
USE_WEBCAM = False  # True for webcam, False for video

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(dirVideo)

# https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps=5.0
frame_width = 680
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size) 
ContFrames=0
ContDetected=0
ContNoDetected=0

while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        break

    # 1️⃣ SLAM – camera pose estimation
    camera_pose = slam.track(img)
    # Example pose: (x, y, theta)

    # Convert pose to SLAM state (simple logic)
    if camera_pose is None:
        slam_state = "LOST"
    else:
        slam_state = "TRACKING"


    # 2️⃣ YOLO detection
    detections = []

    if slam_state == "TRACKING":
        results = model(img, conf=0.4, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = class_list[class_id] + f" {conf:.2f}"

            crop = img[y1:y2, x1:x2]

            print(class_id)
            print(label)

            detections.append([x1, y1, x2, y2, conf])

    else:
        detections = []


    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # 3️⃣ SORT tracking
    tracks = tracker.update(detections)

    # 4️⃣ Draw tracked boxes + alert + SLAM info
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        # 🔴 ALERT LOGIC (only once per object)
        if track_id not in alerted_ids:
            print(f"Traffic Sign Detected | ID: {track_id} | Pose: {camera_pose} | SLAM: {slam_state}")
            alerted_ids.add(track_id)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Display tracking ID
        cv2.putText(
            img,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Display SLAM pose
        cv2.putText(
            img,
            f"Pose: {camera_pose}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 5️⃣ Display frame
    img_show = cv2.resize(img, (frame_width, frame_height))
    cv2.imshow("YOLO + SORT + SLAM", img_show)

    # 6️⃣ Save frame
    video_writer.write(img)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Time limit
    if time.time() - TimeIni > TimeLimit:
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\nProcessing finished")
print("Time in seconds:", time.time() - TimeIni)