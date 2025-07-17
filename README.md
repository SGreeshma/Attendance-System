# AUTOMATED ATTENDANCE USING FACE RECOGNITION
A real-time attendance system using Python and OpenCV for face detection and recognition. Attendance is logged in CSV format for easy tracking.

## About
Automates attendance through face recognition using webcam. Detects and recognizes faces, logs date and time in CSV files.

## Tech Stack
- Python
- OpenCV
- Pandas
- Haar Cascade (Face Detection)
- LBPH (Face Recognition)
- Tkinter / Flask (Optional UI)

## Features
- Real-time face detection
- Face recognition with LBPH
- Logs name, date, time
- Stores CSV files per day
- GUI or Web-based interface (optional)

## Project Structure
attendance-system/
├── backend/
│ ├── app.py
│ ├── face_capture.py    # Capture student images
│ ├── train_model.py     # Train LBPH model
│ ├── recognize_face.py  # Real-time recognition
├── data/
│ ├── pics/              # Collected student images
│ └── rollno_name        # Saved face model
├── attendanceRecords/
│ └── 2025-07-17.csv
├── frontend/
│   ├── main_gui.py      # Tkinter-based dashboard

<pre> ```bash attendance-system/ ├── backend/ │ ├── app.py # Flask server │ ├── face_capture.py # Capture student images │ ├── train_model.py # Train LBPH model │ ├── recognize_face.py # Real-time recognition ├── data/ │ ├── pics/ # Collected student images │ └── rollno_name # Saved face model ├── attendanceRecords/ │ └── 2025-07-17.csv # Daily CSV logs ├── frontend/ │ ├── main_gui.py # Tkinter-based dashboard ``` </pre>

## How to Use
### 1. Login
Enter your credentials to access the system.
### 2. Dashboard Functions
- **Capture Student Images** – Capture student face data via webcam.
- **Train Model** – Train the face recognizer on collected data.
- **Start Attendance** – Detect faces and mark attendance automatically.
- **View Enrolled Students** – Display all enrolled students.
- **View Attendance Logs** – Show/download daily CSV reports.
### 3. Attendance Logs
Logs are stored day-wise in:
attendance_logs/
├── 2025-07-17.csv
├── 2025-07-18.csv
...

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Greeshma Somarouthu
Email: somarouthugreeshma
Location: Hyderabad, India

## License
This project is licensed under the MIT License.
