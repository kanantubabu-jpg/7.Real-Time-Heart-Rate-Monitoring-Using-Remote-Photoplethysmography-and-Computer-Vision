import cv2


class EyeDetector:

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect(self, frame, face):
        x, y, w, h = face
        upper_face_height = max(1, int(h * 0.65))

        roi_gray = cv2.cvtColor(
            frame[y:y + upper_face_height, x:x + w],
            cv2.COLOR_BGR2GRAY
        )

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(int(w * 0.08), int(h * 0.04)),
            maxSize=(int(w * 0.45), int(h * 0.25)),
        )

        filtered_eyes = []
        for ex, ey, ew, eh in eyes:
            if eh == 0:
                continue
            ratio = ew / eh
            if ratio < 0.7 or ratio > 4.0:
                continue
            filtered_eyes.append((ex, ey, ew, eh))

        if len(filtered_eyes) > 2:
            filtered_eyes = sorted(filtered_eyes, key=lambda rect: rect[0])[:2]

        return filtered_eyes

    @staticmethod
    def estimate_eye_regions(face):
        """Fallback eye positions from face geometry when Haar detection fails."""
        x, y, w, h = face
        return [
            (int(w * 0.12), int(h * 0.26), int(w * 0.30), int(h * 0.14)),
            (int(w * 0.58), int(h * 0.26), int(w * 0.30), int(h * 0.14)),
        ]

    def get_eye_state(self, eyes, face, estimated=False):
        if estimated:
            return 'Estimated'
        if len(eyes) >= 2:
            return 'Both Visible'
        if len(eyes) == 1:
            return 'One Visible'
        return 'Not Visible'
