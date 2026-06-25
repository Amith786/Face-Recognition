import cv2

facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        print("Camera Error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()