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

## Screenshots
### Login portal
<img width="498" height="608" alt="image" src="https://github.com/user-attachments/assets/6d3008cb-c9c2-488d-b95e-1cd32a522afa" />

### Dashboard
<img width="1063" height="563" alt="image" src="https://github.com/user-attachments/assets/368706b3-1ec3-41c5-9441-a6ffe80adeed" />

### Details to collect student data
<img width="491" height="608" alt="image" src="https://github.com/user-attachments/assets/da87514f-6782-4004-a74d-120ed536eccf" />
<img width="491" height="609" alt="image" src="https://github.com/user-attachments/assets/614fb340-7c26-4256-bfc9-1c45fbba7d3c" />

### Capturing 
<img width="650" height="560" alt="image" src="https://github.com/user-attachments/assets/7ff68605-6246-4040-b7e4-ff480a3e642b" />

### Recognition of multiple faces simultaneously
<img width="525" height="576" alt="image" src="https://github.com/user-attachments/assets/64fe0cc4-e078-4037-9224-d5846f969565" />

### Attendance log
<img width="411" height="520" alt="image" src="https://github.com/user-attachments/assets/acb4f07b-2f20-4afd-8bc1-156f264022b0" />

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
