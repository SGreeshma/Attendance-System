import cv2
import os

data_dir = "pics"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_images(student_name, student_roll, max_images=150):
    if not student_name or not student_roll:
        raise ValueError("Student name and roll number are required.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Camera could not be opened.")

    count = 0
    person_path = os.path.join(data_dir, f"{student_roll}_{student_name}")
    os.makedirs(person_path, exist_ok=True)

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_path, f"{count}.jpg"), face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Capturing Faces - Press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        raise Exception("No faces captured.")
    
    print(f"✅ Captured {count} images for {student_name} ({student_roll}).")
    return count
