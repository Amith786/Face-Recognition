import cv2
import os
import json

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

user_id = input("Enter User ID: ")
user_name = input("Enter User Name: ")

# Save ID and Name automatically
if os.path.exists("users.json"):
    with open("users.json", "r") as f:
        users = json.load(f)
else:
    users = {}

users[user_id] = user_name

with open("users.json", "w") as f:
    json.dump(users, f, indent=4)

cam = cv2.VideoCapture(0)

count = 0

if not os.path.exists("dataset"):
    os.makedirs("dataset")

print("Look at the camera...")
print("Capturing 100 images...")

while True:

    ret, img = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        cv2.imwrite(
            f"dataset/User.{user_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Images: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture", img)

    k = cv2.waitKey(1)

    if k == 27:  # ESC key
        break

    if count >= 100:
        break

cam.release()
cv2.destroyAllWindows()

print(f"Dataset created for {user_name}")