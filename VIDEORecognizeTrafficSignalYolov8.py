# ================= IMPORTS =================
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tracker.sort import Sort

# ================= PARAMETERS =================
dirVideo = "traffic-sign-to-test.mp4"
dirnameYolo = "bestDetectTrafficSign.pt"
outputVideo = "demonstration.mp4"

# ================= LOAD MODELS =================
cnn_model = load_model("model.h5", compile=False)
yolo = YOLO(dirnameYolo)

# ================= CLASS LABELS (FULL 43) =================
classes = {
    1:'Speed limit (20km/h)', 
    2:'Speed limit (30km/h)', 
    3:'Speed limit (50km/h)',
    4:'Speed limit (60km/h)', 
    5:'Speed limit (70km/h)', 
    6:'Speed limit (80km/h)',
    7:'End of speed limit (80km/h)', 
    8:'Speed limit (100km/h)',
    9:'Speed limit (120km/h)', 
    10:'No passing',
    11:'No passing veh over 3.5 tons', 
    12:'Right-of-way at intersection',
    13:'Priority road', 
    14:'Yield', 
    15:'Stop', 
    16:'No vehicles',
    17:'Veh > 3.5 tons prohibited', 
    18:'No entry',
    19:'General caution', 
    20:'Dangerous curve left',
    21:'Dangerous curve right', 
    22:'Double curve',
    23:'Bumpy road', 
    24:'Slippery road',
    25:'Road narrows on the right', 
    26:'Road work',
    27:'Traffic signals', 
    28:'Pedestrians',
    29:'Children crossing', 
    30:'Bicycles crossing',
    31:'Beware of ice/snow', 
    32:'Wild animals crossing',
    33:'End speed + passing limits', 
    34:'Turn right ahead',
    35:'Turn left ahead', 
    36:'Ahead only',
    37:'Go straight or right', 
    38:'Go straight or left',
    39:'Keep right', 
    40:'Keep left',
    41:'Roundabout mandatory', 
    42:'End of no passing',
    43:'End no passing veh > 3.5 tons'
}

def DetectTrafficSignWithYolov8(img, yolo_model, class_list):
    detections = []
    crops = []
    labels = []
    boxes_list = []

    results = yolo_model.predict(img, conf=0.5, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(class_ids)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = float(confs[i])
            cls_id = class_ids[i]

            if conf < 0.4:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = f"{class_list[cls_id]} ({conf:.2f})"

            detections.append([x1, y1, x2, y2, conf])
            crops.append(crop)
            labels.append(label)
            boxes_list.append([x1, y1, x2, y2])

            print(f"[YOLO] {label}")

    return detections, crops, labels, boxes_list



# ================= SORT TRACKER =================
tracker = Sort()
alerted_ids = set()

last_print_time = {}
PRINT_INTERVAL = 0.5  # seconds (avoid console spam)


# ================= SLAM MODULE =================
class VisualSLAM:
    def __init__(self):
        self.orb = cv2.ORB_create(2000)
        self.prev_des = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is None or des is None:
            self.prev_des = des
            return "INIT"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.prev_des, des)
        self.prev_des = des

        return "TRACKING" if len(matches) > 25 else "LOST"

slam = VisualSLAM()

# ================= DISTANCE ESTIMATION =================
def estimate_distance(box_height, ref_height=120, ref_distance=10):
    if box_height == 0:
        return None
    return round((ref_height / box_height) * ref_distance, 2)

# ================= VIDEO SETUP =================
cap = cv2.VideoCapture(dirVideo)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 5.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(outputVideo, fourcc, fps, (frame_width, frame_height))

TimeIni = time.time()
TimeLimit = 1000  # seconds


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    slam_state = slam.update(frame)
    detections, crops, yolo_labels, yolo_boxes = [], [], [], []
    
    if slam_state != "TRACKING":
        print(f"[SLAM] State = {slam_state}, skipping YOLO")

    if slam_state == "TRACKING":
        detections, crops, yolo_labels, yolo_boxes = DetectTrafficSignWithYolov8(
            frame, yolo, yolo.model.names
        )



    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracks = tracker.update(detections)

    # ===== PROCESS TRACKED OBJECTS =====
    for trk in tracks:
        x1, y1, x2, y2, track_id = trk.astype(int)
        track_box = [x1, y1, x2, y2]

        # ---- Match SORT track to YOLO detection ----
        best_iou = 0
        best_idx = -1

        for idx, ybox in enumerate(yolo_boxes):
            score = iou(track_box, ybox)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx == -1:
            continue

        crop = crops[best_idx]
        yolo_label = yolo_labels[best_idx]
        yolo_box = yolo_boxes[best_idx]

        # ---- CNN CLASSIFICATION ----
        crop_resized = cv2.resize(crop, (32, 32))
        crop_input = np.expand_dims(crop_resized, axis=0)
        preds = cnn_model.predict(crop_input, verbose=0)
        cls_id = np.argmax(preds) + 1
        cnn_label = classes.get(cls_id, "Unknown")

        # ---- Distance (use YOLO box height) ----
        distance = estimate_distance(yolo_box[3] - yolo_box[1])

        # ---- Console debug ----
        current_time = time.time()
        if (
            track_id not in last_print_time or
            current_time - last_print_time[track_id] > PRINT_INTERVAL
        ):
            print(
                f"[TRACK {track_id}] YOLO={yolo_label} | CNN={cnn_label} | "
                f"Distance={distance}m | SLAM={slam_state}"
            )
            last_print_time[track_id] = current_time

        if track_id not in alerted_ids:
            print(f"ALERT: {cnn_label} detected for the first time")
            alerted_ids.add(track_id)

        # ---- Draw ----
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(
            frame,
            f"YOLO: {yolo_label} | CNN: {cnn_label} | {distance}m",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )



    # ===== SLAM STATUS =====
    cv2.putText(frame, f"SLAM: {slam_state}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display + Save
    cv2.imshow("Traffic Sign System", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - TimeIni > TimeLimit:
        break

# ================= CLEANUP =================
cap.release()
video_writer.release()
cv2.destroyAllWindows()
