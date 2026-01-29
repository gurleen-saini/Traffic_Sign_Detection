import cv2


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