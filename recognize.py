print("="*40)
print("FACE RECOGNITION SYSTEM")
print("Press Q to Exit")
print("="*40)
import cv2
import json

# Load user names
with open("users.json", "r") as f:
    users = json.load(f)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

print("Face Recognition Started...")
print("Press Q to Exit")

while True:

    ret, img = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        user_id, confidence = recognizer.predict(face_roi)

        confidence_percent = round(100 - confidence)

        if confidence < 80:
            name = users.get(str(user_id), "Unknown")
            color = (0, 255, 0)
        else:
            name = "Unknown"
            confidence_percent = 0
            color = (0, 0, 255)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            img,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            img,
            f"{confidence_percent}%",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    cv2.putText(
        img,
        "Face Recognition System",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.imshow("Face Recognition System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()