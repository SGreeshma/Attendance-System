import cv2
import os
import numpy as np
from PIL import Image
import pickle

data_dir = "pics"
trainer_dir = "trainer"
os.makedirs(trainer_dir, exist_ok=True)

def train_model():
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("⚠️ No student data found in 'pics/' folder. Please capture images first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_mapping = {}
    current_id = 0

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        identity = folder.strip()  # e.g., "2101_John"

        if identity not in label_mapping:
            label_mapping[identity] = current_id
            current_id += 1

        label_id = label_mapping[identity]

        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, img_name)
            try:
                image = Image.open(img_path).convert('L')
                img_np = np.array(image, 'uint8')
                faces.append(img_np)
                labels.append(label_id)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    if not faces:
        print("No valid training images found.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(os.path.join(trainer_dir, "trainer.yml"))
    with open(os.path.join(trainer_dir, "labels.pkl"), "wb") as f:
        pickle.dump(label_mapping, f)

    print("✅ Model trained successfully with roll numbers and names.")
