from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from api.video_utils import read_video
from core.pipeline import TrafficSignPipeline

# ================= APP INIT =================
app = FastAPI(
    title="Traffic Sign Detection API",
    description="YOLOv8 + CNN + SLAM + SORT Traffic Sign Detection",
    version="1.0.0"
)

# ================= PIPELINE INIT =================
# This loads:
# - YOLOv8 model
# - CNN model
# - SLAM
# - SORT tracker
pipeline = TrafficSignPipeline()

# ================= ROOT =================
@app.get("/")
def root():
    return {
        "message": "Traffic Sign Detection API is running",
        "endpoints": {
            "video_detection": "/detect/video",
            "docs": "/docs"
        }
    }

# ================= VIDEO UPLOAD =================
@app.post("/detect/video")
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
        frame_results = pipeline.process_frame(frame)

        # Attach frame number
        for r in frame_results:
            r["frame"] = frame_id
            all_results.append(r)

    # ---- Cleanup ----
    os.remove(video_path)

    return {
        "total_detections": len(all_results),
        "results": all_results
    }
