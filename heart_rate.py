import time
import cv2
import numpy as np
from signal_processing import (
    estimate_bpm_from_rgb,
    signal_quality_score_rgb,
    reject_bpm_outlier,
)


class HeartRateMonitor:

    MIN_SAMPLES = 45
    MIN_DURATION_SEC = 2.5
    MAX_WINDOW = 150
    MIN_FS = 8
    SMOOTH_WINDOW = 5

    def __init__(self):
        self.rgb_values = []
        self.timestamps = []
        self.bpm_history = []
        self.last_bpm = None

    def reset(self):
        self.rgb_values = []
        self.timestamps = []
        self.bpm_history = []
        self.last_bpm = None

    @staticmethod
    def extract_rgb_means(roi):
        if roi is None or roi.size == 0:
            return None
        return [
            float(np.median(roi[:, :, 0])),
            float(np.median(roi[:, :, 1])),
            float(np.median(roi[:, :, 2])),
        ]

    @staticmethod
    def _skin_patches(frame, face, eyes):
        x, y, w, h = face
        height, width = frame.shape[:2]
        patches = []

        for ex, ey, ew, eh in eyes:
            regions = [
                (x + ex, y + ey + eh, ew, max(4, int(eh * 0.50))),
                (x + ex, max(0, y + ey - int(eh * 0.30)), ew, max(4, int(eh * 0.25))),
            ]

            for abs_x, abs_y, patch_w, patch_h in regions:
                abs_x2 = min(width, abs_x + patch_w)
                abs_y2 = min(height, abs_y + patch_h)
                abs_x = max(0, abs_x)
                abs_y = max(0, abs_y)
                if abs_x2 <= abs_x or abs_y2 <= abs_y:
                    continue
                patch = frame[abs_y:abs_y2, abs_x:abs_x2]
                if patch.size > 0:
                    patches.append(patch)

        return patches

    @staticmethod
    def extract_eye_roi(frame, face, eyes):
        """Sample periorbital skin (not the pupil) for rPPG."""
        patches = HeartRateMonitor._skin_patches(frame, face, eyes)
        if not patches:
            return None

        if len(patches) == 1:
            return patches[0]

        target_h = max(4, min(p.shape[0] for p in patches))
        target_w = max(4, min(p.shape[1] for p in patches))
        resized = [
            cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for patch in patches
        ]
        return np.mean(resized, axis=0).astype(np.uint8)

    @staticmethod
    def eye_roi_bounds(frame, face, eyes):
        if not eyes:
            return None

        x, y, w, h = face
        height, width = frame.shape[:2]
        left = width
        top = height
        right = 0
        bottom = 0

        for ex, ey, ew, eh in eyes:
            pad_x = int(ew * 0.15)
            abs_x = max(0, x + ex - pad_x)
            abs_y = max(0, y + ey - int(eh * 0.30))
            abs_x2 = min(width, x + ex + ew + pad_x)
            abs_y2 = min(height, y + ey + eh + int(eh * 0.50))
            left = min(left, abs_x)
            top = min(top, abs_y)
            right = max(right, abs_x2)
            bottom = max(bottom, abs_y2)

        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def update(self, roi, face_visible=True):
        if not face_visible:
            self.reset()
            return

        if roi is None:
            return

        rgb = self.extract_rgb_means(roi)
        if rgb is None:
            return

        self.rgb_values.append(rgb)
        self.timestamps.append(time.time())

        if len(self.rgb_values) > 300:
            self.rgb_values.pop(0)
            self.timestamps.pop(0)

    def _estimate_from_buffer(self):
        if len(self.rgb_values) < self.MIN_SAMPLES:
            return None

        window_size = min(len(self.rgb_values), self.MAX_WINDOW)
        rgb = np.array(self.rgb_values[-window_size:], dtype=np.float64)
        time_window = np.array(self.timestamps[-window_size:])

        duration = time_window[-1] - time_window[0]
        if duration < self.MIN_DURATION_SEC:
            return None

        fs = len(rgb) / duration
        if fs < self.MIN_FS:
            return None

        return estimate_bpm_from_rgb(rgb, fs, prev_bpm=self.last_bpm)

    def get_bpm(self):
        bpm = self._estimate_from_buffer()
        bpm = reject_bpm_outlier(bpm, self.bpm_history, max_jump=10)

        if bpm is not None:
            self.bpm_history.append(bpm)
            if len(self.bpm_history) > 10:
                self.bpm_history.pop(0)

            window = self.bpm_history[-self.SMOOTH_WINDOW:]
            self.last_bpm = int(round(np.median(window)))

        return self.last_bpm

    def get_status(self, face_visible):
        if not face_visible:
            self.reset()
            return {'bpm': None, 'quality': 'No Face', 'samples': 0}

        sample_count = len(self.rgb_values)
        if sample_count < self.MIN_SAMPLES // 2:
            return {'bpm': None, 'quality': 'Collecting', 'samples': sample_count}

        window_size = min(len(self.rgb_values), self.MAX_WINDOW)
        rgb = np.array(self.rgb_values[-window_size:], dtype=np.float64)
        time_window = np.array(self.timestamps[-window_size:])
        duration = time_window[-1] - time_window[0]
        fs = len(rgb) / duration if duration > 0 else 0

        score = signal_quality_score_rgb(rgb, fs) if fs >= self.MIN_FS else 0.0
        if score >= 0.5:
            quality = 'Good'
        elif score >= 0.25:
            quality = 'Fair'
        else:
            quality = 'Poor'

        bpm = self.get_bpm()
        return {'bpm': bpm, 'quality': quality, 'samples': sample_count}
