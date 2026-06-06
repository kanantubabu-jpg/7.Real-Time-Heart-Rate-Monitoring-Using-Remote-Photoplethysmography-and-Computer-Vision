import cv2

class FaceDetector:
    def __init__(self):
        self.frontal_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.tracker = None
        self.smoothed_face = None
        self.lost = True

    def _new_tracker(self):
        fallback_factories = []
        if hasattr(cv2, 'legacy'):
            fallback_factories.extend([
                lambda: cv2.legacy.TrackerCSRT_create(),
                lambda: cv2.legacy.TrackerKCF_create(),
                lambda: cv2.legacy.TrackerMOSSE_create(),
            ])
        fallback_factories.extend([
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.TrackerKCF_create(),
            lambda: cv2.TrackerMOSSE_create(),
        ])

        for factory in fallback_factories:
            try:
                tracker = factory()
                if tracker is not None:
                    return tracker
            except Exception:
                continue

        return None

    def _valid_rect(self, rect, frame):
        x, y, w, h = [int(max(0, v)) for v in rect]
        height, width = frame.shape[:2]
        if w < 60 or h < 60:
            return False
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return False
        return True

    def _smooth_face(self, face):
        if self.smoothed_face is None:
            self.smoothed_face = face
            return face

        alpha = 0.7
        x = int(self.smoothed_face[0] * alpha + face[0] * (1 - alpha))
        y = int(self.smoothed_face[1] * alpha + face[1] * (1 - alpha))
        w = int(self.smoothed_face[2] * alpha + face[2] * (1 - alpha))
        h = int(self.smoothed_face[3] * alpha + face[3] * (1 - alpha))
        self.smoothed_face = (x, y, w, h)
        return self.smoothed_face

    def _detect_faces(self, gray):
        faces = []
        frontal = self.frontal_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(frontal) > 0:
            faces.extend(frontal)

        profiles = self.profile_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(profiles) > 0:
            faces.extend(profiles)

        flipped = cv2.flip(gray, 1)
        profiles_flipped = self.profile_detector.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        for x, y, w, h in profiles_flipped:
            faces.append((gray.shape[1] - x - w, y, w, h))

        return faces

    def _init_tracker(self, frame, face):
        self.tracker = self._new_tracker()
        if self.tracker is None:
            return
        try:
            self.tracker.init(frame, tuple(face))
        except Exception:
            self.tracker = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.tracker is not None:
            success, tracked_box = self.tracker.update(frame)
            if success and self._valid_rect(tracked_box, frame):
                self.lost = False
                face = tuple(map(int, tracked_box))
                return [self._smooth_face(face)]
            self.tracker = None

        faces = self._detect_faces(gray)
        if len(faces) == 0:
            self.lost = True
            return []

        face = max(faces, key=lambda rect: rect[2] * rect[3])
        self.lost = False
        face = tuple(map(int, face))
        self._init_tracker(frame, face)
        return [self._smooth_face(face)]
