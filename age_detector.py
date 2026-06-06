import logging
import os
import cv2

logger = logging.getLogger(__name__)

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(21-24)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]


class AgeDetector:

    def __init__(self):
        self.model = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        proto_path = os.path.join(base_dir, "models", "age_deploy.prototxt")
        model_path = os.path.join(base_dir, "models", "age_net.caffemodel")

        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            logger.warning(
                "Age model files missing; age detection disabled. Expected:\n"
                "  %s\n  %s",
                proto_path,
                model_path,
            )
            return

        if os.path.getsize(model_path) < 1024 or os.path.getsize(proto_path) < 64:
            logger.warning(
                "Age model files appear invalid or incomplete; age detection disabled."
            )
            return

        try:
            self.model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        except cv2.error as exc:
            logger.warning("Failed to load age model; age detection disabled: %s", exc)

    @property
    def available(self):
        return self.model is not None

    def predict(self, frame, face):
        if not self.available:
            return "N/A"

        x, y, w, h = face
        face_img = frame[y:y + h, x:x + w]

        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
        )

        self.model.setInput(blob)
        preds = self.model.forward()
        return AGE_BUCKETS[preds[0].argmax()]
