import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
else:
    print("Camera detected")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()