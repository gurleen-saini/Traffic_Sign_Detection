import cv2

def read_video(video_path):
    """
    Generator that yields frames from a video file
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video file")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()


def get_video_info(video_path):
    """
    Returns fps, width, height of the video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height


def draw_bbox(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    """
    Draw bounding box and label on frame
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


