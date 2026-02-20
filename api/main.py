from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import numpy as np

from db.database import SessionLocal, engine
from db import models, schemas
from db.auth import hash_password, verify_password
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session


from api.video_utils import read_video
from core.pipeline import TrafficSignPipeline
# from core.pipeline_rcnn import TrafficSignRCNNPipeline 
# from core.pipeline_yolov7 import TrafficSignYOLOv7Pipeline
# from core.pipeline_ssd import TrafficSignSSDPipeline
from fastapi import WebSocket, WebSocketDisconnect

import cv2
import base64


# ================= APP INIT =================
app = FastAPI(
    title="Traffic Sign Detection API",
    description="YOLOv8 + CNN + SLAM + SORT Traffic Sign Detection  ||  YOLOv8 + Faster R-CNN Traffic Sign Detection",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    print("Initializing database...")
    models.Base.metadata.create_all(bind=engine)
    print("Database ready.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/auth/signup")
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):

    existing = db.query(models.User).filter(
        models.User.email == user.email
    ).first()

    if existing:
        raise HTTPException(status_code=400,
                            detail="Email already exists")

    new_user = models.User(
        email=user.email,
        password=hash_password(user.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created successfully"}

@app.post("/auth/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):

    db_user = db.query(models.User).filter(
        models.User.email == user.email
    ).first()

    if not db_user:
        raise HTTPException(status_code=401,
                            detail="Invalid credentials")

    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401,
                            detail="Invalid credentials")

    return {"message": "Login successful"}

pipeline = None
# rcnn_pipeline = None
# yolov7_pipeline = None
# ssd_pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading YOLOv8 Pipeline...")
        pipeline = TrafficSignPipeline()
    return pipeline


# def get_rcnn_pipeline():
#     global rcnn_pipeline
#     if rcnn_pipeline is None:
#         print("Loading RCNN Pipeline...")
#         # rcnn_pipeline = TrafficSignRCNNPipeline()
#     return rcnn_pipeline


# def get_yolov7_pipeline():
#     global yolov7_pipeline
#     if yolov7_pipeline is None:
#         print("Loading YOLOv7 Pipeline...")
#         # yolov7_pipeline = TrafficSignYOLOv7Pipeline()
#     return yolov7_pipeline


# def get_ssd_pipeline():
#     global ssd_pipeline
#     if ssd_pipeline is None:
#         print("Loading SSD Pipeline...")
#         # ssd_pipeline = TrafficSignSSDPipeline()
#     return ssd_pipeline

# ================= ROOT =================
@app.get("/")
def root():
    return {
        "message": "Traffic Sign Detection API is running",
        "endpoints": {
            "video_detection": "/detect/video",
            # "rcnn_video": "/detect/video/rcnn",
            "docs": "/docs"
        }
    }

# ================= VIDEO UPLOAD =================
@app.post("/detect/video/yolo")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video and run full traffic sign detection pipeline.
    Returns per-frame detection results.
    """

    # ---- Save uploaded video temporarily ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    all_results = []
    frame_id = 0

    # ---- Process video frame-by-frame ----
    for frame in read_video(video_path):
        frame_id += 1

        # FULL LOGIC EXECUTION (same as old script)
        frame_results = get_pipeline().process_frame(frame)

        # Attach frame number
        for r in frame_results:
            r["frame"] = frame_id
            r["model"] = "yolov8"
            all_results.append(r)

    # ---- Cleanup ----
    os.remove(video_path)

    return {
        "model": "yolov8",
        "total_detections": len(all_results),
        "results": all_results
    }

# # ================= RCNN VIDEO =================
# @app.post("/detect/video/rcnn")
# async def detect_video_rcnn(file: UploadFile = File(...)):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await file.read())
#         video_path = tmp.name

#     all_results = []
#     frame_id = 0

#     for frame in read_video(video_path):
#         frame_id += 1
#         frame_results = get_rcnn_pipeline().process_frame(frame)

#         for r in frame_results:
#             r["frame"] = frame_id
#             all_results.append(r)

#     os.remove(video_path)

#     return {
#         "model": "faster_rcnn",
#         "total_detections": len(all_results),
#         "results": all_results
#     }

# # ================= SSD VIDEO =================
# # ================= SSD VIDEO =================


# @app.post("/detect/video/ssd")
# async def detect_video_ssd(file: UploadFile = File(...)):

    # # ---- Save uploaded video temporarily ----
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    #     tmp.write(await file.read())
    #     video_path = tmp.name

    # all_results = []
    # frame_id = 0

    # # ---- Process video frame-by-frame ----
    # for frame in read_video(video_path):
    #     frame_id += 1
    #     frame_results = get_ssd_pipeline().process_frame(frame)

    #     for r in frame_results:
    #         r["frame"] = frame_id
    #         all_results.append(r)

#     # ---- Cleanup ----
#     os.remove(video_path)

#     return {
#         "model": "ssd",
#         "total_detections": len(all_results),
#         "results": all_results
#     }

# @app.post("/detect/video/yolov7")
# async def detect_video_yolov7(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await file.read())
#         video_path = tmp.name

#     all_results = []
#     frame_id = 0

#     for frame in read_video(video_path):
#         frame_id += 1
#         frame_results = get_yolov7_pipeline().process_frame(frame)

    #     for r in frame_results:
    #         r["frame"] = frame_id
    #         r["model"] = "yolov7"
    #         all_results.append(r)

    # os.remove(video_path)

    # return {
    #     "model": "yolov7",
    #     "total_detections": len(all_results),
    #     "results": all_results
    # }


def draw_boxes(frame, results):

    for r in results:

        bbox = r.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        cnn_label = r.get("cnn_label", "")
        distance = r.get("distance", "")

        text = f"{cnn_label} | {distance}m"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    return frame


def filter_results(results):

    filtered = []

    for r in results:

        # ----- YOLO CONF -----
        yolo_label = r.get("yolo_label", "")
        try:
            yolo_conf = float(
                yolo_label.split("(")[-1].replace(")", "")
            )
            if yolo_conf < 0.75:
                continue
        except:
            continue

        # ----- CNN CONF -----
        cnn_conf = r.get("cnn_confidence", 0)
        if cnn_conf < 0.75:
            continue

        filtered.append(r)

    return filtered

@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()

            # Convert bytes -> image
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = get_pipeline().process_frame(frame)

            # FILTER LOW CONFIDENCE
            results = filter_results(results)

            print(results)

            frame = draw_boxes(frame, results)

            # Encode image
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode()

            await websocket.send_json({
                "image": frame_base64,
                "detections": results
            })

    except WebSocketDisconnect:
        print("Mobile disconnected")
