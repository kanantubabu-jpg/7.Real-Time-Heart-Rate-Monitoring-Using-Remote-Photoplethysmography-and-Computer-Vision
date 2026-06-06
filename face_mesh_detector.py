import cv2
import numpy as np

class FaceMeshDetector:
    def __init__(self):
        pass

    def get_forehead_roi(self, frame, face):
        if face is None:
            return None, frame

        x, y, w, h = face
        forehead_height = max(10, int(h * 0.25))
        forehead_width = max(10, int(w * 0.5))
        fx = x + int((w - forehead_width) / 2)
        fy = y + int(h * 0.08)

        if fy + forehead_height > frame.shape[0] or fx + forehead_width > frame.shape[1]:
            return None, frame

        roi = frame[fy:fy + forehead_height, fx:fx + forehead_width]
        cv2.rectangle(frame, (fx, fy), (fx + forehead_width, fy + forehead_height), (0, 255, 0), 2)

        return roi, frame
