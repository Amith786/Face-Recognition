import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "dataset"

face_samples = []
ids = []

for image_name in os.listdir(path):

    image_path = os.path.join(path, image_name)

    gray_img = Image.open(image_path).convert('L')

    img_numpy = np.array(gray_img, 'uint8')

    user_id = int(image_name.split(".")[1])

    face_samples.append(img_numpy)
    ids.append(user_id)

print("Training...")

recognizer.train(face_samples, np.array(ids))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save("trainer/trainer.yml")

print("Training Completed!")