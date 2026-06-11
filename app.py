import time
import logging
import threading
import cv2
import os
from flask import Flask, render_template, Response, jsonify

from age_detector import AgeDetector
from face_detector import FaceDetector
from eye_detector import EyeDetector
from heart_rate import HeartRateMonitor

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

capture_backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0


cap = None

if os.environ.get("RENDER") is None:
    cap = cv2.VideoCapture(0, capture_backend)
face_detector = FaceDetector()
eye_detector = EyeDetector()
age_detector = AgeDetector()
hr_monitor = HeartRateMonitor()
latest_frame = None
latest_status = {}
data_lock = threading.Lock()

previous_face_visible = False
previous_eye_status = 'Not Visible'


def draw_overlay(frame, face_status, eye_status, age_text, bpm_text, quality_text):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (8, 8), (frame.shape[1] - 8, 132), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    labels = [
        f'Face: {face_status}',
        f'Eyes: {eye_status}',
        f'Age: {age_text}',
        f'Heart Rate: {bpm_text}',
        f'Signal Quality: {quality_text}',
    ]

    y = 32
    for label in labels:
        cv2.putText(
            frame,
            label,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def format_bpm(bpm):
    return f'{bpm} BPM' if bpm is not None else 'Measuring...'


def capture_loop():
    global latest_frame, latest_status, previous_face_visible, previous_eye_status

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning('Webcam read failed.')
            time.sleep(0.1)
            continue

        faces = face_detector.detect(frame)
        face_visible = len(faces) > 0
        face_status = 'Detected' if face_visible else 'Lost'
        age_text = 'Unknown'
        eye_status = 'Not Visible'
        quality_text = 'Unknown'
        bpm_text = 'Measuring...'
        roi = None
        eyes = []
        eyes_estimated = False

        if face_visible:
            face = faces[0]
            age_text = age_detector.predict(frame, face)
            eyes = eye_detector.detect(frame, face)
            if len(eyes) == 0:
                eyes = eye_detector.estimate_eye_regions(face)
                eyes_estimated = len(eyes) > 0
            eye_status = eye_detector.get_eye_state(eyes, face, estimated=eyes_estimated)
            if len(eyes) > 0:
                roi = HeartRateMonitor.extract_eye_roi(frame, face, eyes)

            if not previous_face_visible:
                logging.info('Face Found')
            if eye_status == 'Not Visible' and previous_eye_status != 'Not Visible':
                logging.warning('Eye Detection Failed')

            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 128, 0),
                    2,
                )

            roi_bounds = HeartRateMonitor.eye_roi_bounds(frame, face, eyes)
            if roi_bounds is not None:
                left, top, right, bottom = roi_bounds
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        else:
            if previous_face_visible:
                logging.info('Face Lost')

        if not face_visible:
            hr_monitor.update(None, face_visible=False)
        elif roi is not None:
            hr_monitor.update(roi, face_visible=True)

        status = hr_monitor.get_status(face_visible)
        bpm = status['bpm']
        latest_status = status
        quality_text = status['quality']
        bpm_text = format_bpm(bpm)

        if bpm is not None:
            bpm_text = f'{bpm} BPM'

        draw_overlay(frame, face_status, eye_status, age_text, bpm_text, quality_text)

        with data_lock:
            latest_frame = frame.copy()
            latest_status = status.copy()

        previous_face_visible = face_visible
        previous_eye_status = eye_status


def gen_frames():
    while True:
        with data_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/bpm_data')
def bpm_data():
    with data_lock:
        bpm = latest_status.get('bpm')
    return jsonify({'bpm': bpm})


@app.route('/status')
def status():
    with data_lock:
        return jsonify(latest_status)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    if not cap.isOpened():
        logging.error('Error: Cannot open webcam.')
    else:
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        app.run(host='0.0.0.0', port=5000, threaded=True)
