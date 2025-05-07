import cv2
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import mediapipe as mp

trainer_dir = "trainer"

EAR_THRESHOLD = 0.18
CONSEC_FRAMES = 3

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def mark_attendance(roll_number):
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Folder to store attendance files
    records_dir = "attendanceRecords"
    os.makedirs(records_dir, exist_ok=True)

    # File path for today's attendance
    file_path = os.path.join(records_dir, f"{date_str}.xlsx")

    # Load or initialize DataFrame
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["Roll Number", "Time"])

    # Prevent duplicate entries
    if roll_number in df["Roll Number"].values:
        print(f"⚠️ {roll_number} has already marked attendance today.")
        return

    # Add new entry
    new_entry = pd.DataFrame([{
        "Roll Number": roll_number,
        "Time": time_str
    }])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save to Excel
    df.to_excel(file_path, index=False)

    print(f"✅ Attendance marked for {roll_number} at {time_str}.")


def compute_ear(landmarks, indices, img_w, img_h):
    coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in indices]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    if C == 0:
        return 1.0
    return (A + B) / (2.0 * C)

def recognize_with_blink():
    trainer_path = os.path.join(trainer_dir, "trainer.yml")
    label_path = os.path.join(trainer_dir, "labels.pkl")

    if not os.path.exists(trainer_path) or not os.path.exists(label_path):
        raise FileNotFoundError("⚠️ Model not trained yet. Please train the model before starting recognition.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    with open(label_path, "rb") as f:
        label_mapping = pickle.load(f)

    id_to_roll = {v: k for k, v in label_mapping.items()}

    marked_faces = set()
    last_recognized_time = 0
    last_recognized_text = ""
    last_recognized_color = (0, 0, 0)

    blink_counter = 0
    blink_confirmed = False

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Blink detection with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        ear = 1.0
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            left_ear = compute_ear(landmarks, LEFT_EYE, w, h)
            right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0 if left_ear and right_ear else 1.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    blink_confirmed = True
                blink_counter = 0

        for (x, y, w_box, h_box) in faces:
            face_roi = gray[y:y+h_box, x:x+w_box]
            label, confidence = recognizer.predict(face_roi)
            roll_number = id_to_roll.get(label, "Unknown")

            if confidence < 60 and blink_confirmed:
                if roll_number not in marked_faces:
                    mark_attendance(roll_number)
                    marked_faces.add(roll_number)
                last_recognized_text = f"{roll_number} ({confidence:.2f})"
                last_recognized_color = (0, 255, 0)
                last_recognized_time = datetime.now().timestamp()
                blink_confirmed = False

            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), last_recognized_color, 2)
        # Show last recognized text for 3 seconds
        if last_recognized_text and (datetime.now().timestamp() - last_recognized_time < 6):
            cv2.putText(frame, last_recognized_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, last_recognized_color, 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Face + Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()