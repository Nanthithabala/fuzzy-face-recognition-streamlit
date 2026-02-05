import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_PATH)

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face_region = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(face_region)
    eye_area = sum([ew * eh for (_, _, ew, eh) in eyes]) if len(eyes) else 0

    nose_area = 0.15 * w * h
    lip_area = 0.15 * w * h
    jaw_width = w

    features = np.array([eye_area, lip_area, nose_area, jaw_width], dtype=float)
    return features / np.max(features)
