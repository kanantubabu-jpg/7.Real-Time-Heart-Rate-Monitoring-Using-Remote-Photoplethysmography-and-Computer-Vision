<img width="1912" height="996" alt="Screenshot 2026-06-06 205940" src="https://github.com/user-attachments/assets/2ad2d7e6-c261-46dc-9df2-a60e782849fe" />

# 7.Real-Time-Heart-Rate-Monitoring-Using-Remote-Photoplethysmography-and-Computer-Vision
❤️ Real-Time Heart Rate Monitoring Using Remote Photoplethysmography and Computer Vision
Overview

This project implements a real-time heart rate monitoring system using Remote Photoplethysmography (rPPG) and Computer Vision techniques. Unlike traditional heart rate sensors that require physical contact, this system estimates a person's heart rate remotely through a webcam by analyzing subtle color variations in facial skin caused by blood circulation.

The application captures live video, detects and tracks the face, extracts skin-region color signals, processes them using signal filtering techniques, and calculates the heart rate in beats per minute (BPM).

Features
Real-time heart rate estimation using a webcam
Contactless monitoring using Remote PPG (rPPG)
Face detection and tracking with MediaPipe
Signal processing and noise reduction
BPM calculation and visualization
Live waveform plotting
Web-based interface using Flask
Lightweight and easy to run on standard computers
Technologies Used
Python
OpenCV
MediaPipe
NumPy
SciPy
Matplotlib
Flask
Libraries
opencv-python
mediapipe
numpy
scipy
matplotlib
Flask
Working Principle
Capture live video from the webcam.
Detect the face using MediaPipe Face Detection.
Extract the forehead or facial skin region.
Calculate average RGB color values from the selected region.
Apply signal processing techniques to remove noise.
Analyze periodic changes caused by blood flow.
Estimate heart rate in BPM.
Display results through a Flask web application.
Project Structure
heart-rate-monitor/
│
├── app.py
├── camera.py
├── signal_processing.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/yourusername/heart-rate-monitor.git
cd heart-rate-monitor

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Open your browser and visit:

http://127.0.0.1:5000
Applications
Telemedicine
Remote Patient Monitoring
Healthcare Systems
Fitness Tracking
Smart Health Devices
Research in Biomedical Signal Processing
Future Enhancements
Multi-person heart rate detection
Oxygen saturation (SpO₂) estimation
Mobile application integration
Deep Learning-based rPPG models
Cloud-based health monitoring dashboard
Conclusion

This project demonstrates how computer vision and signal processing can be combined to perform contactless heart rate monitoring using a standard webcam. By leveraging Remote Photoplethysmography (rPPG), the system provides a low-cost, non-invasive solution for real-time health monitoring and biomedical research.
