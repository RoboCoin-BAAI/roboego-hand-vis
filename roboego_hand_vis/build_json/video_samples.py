"""Count frames reported by video containers across all episodes of a camera."""
import math


def video_sample_count(paths):
    import cv2

    total = 0
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return None
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(count) or count <= 0:
                return None
            total += int(count)
        finally:
            capture.release()
    return total
