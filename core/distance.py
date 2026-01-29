
# ================= DISTANCE ESTIMATION =================
def estimate_distance(box_height, ref_height=120, ref_distance=10):
    if box_height == 0:
        return None
    return round((ref_height / box_height) * ref_distance, 2)