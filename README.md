AUTOMATED ATTENDANCE USING FACE RECOGNITION
A real-time attendance system that uses OpenCV and Python for face detection and recognition using the LBPH algorithm that detects multiple faces simultaneously at a time. Attendance is stored in CSV format for easy tracking.

🚀 Features
- Real-time face detection using Haar Cascade
- Face recognition with LBPH algorithm
- Attendance logging with name, date, and time
- Stores logs in Excel-compatible CSV format
- Optional: GUI using Tkinter (or web-based UI)

Technologies Used
- Python
- OpenCV
- Pandas
- Haar Cascade (for face detection)
- LBPH (for face recognition)

Attendance Format
Name	Date	Time
Greeshma	2025-07-17	09:02:33

How to Use the Attendance System
1️. Login
Enter your username and password on the login screen.
After successful authentication, the Dashboard appears.

2. Dashboard Functionalities
The dashboard provides the following main options:
2.1 Capture Student Images
    Click this to collect facial data for each student.
    The system uses your webcam to capture multiple images of the student.
    Each student's images are saved.
2.2 Train Model
    After capturing images, click Train Model.
    This will process the collected images and train the face recognition model.
    A trained model file is saved for future use.
2.3 Start Attendance
    Click to begin the real-time face recognition process.
    The system will:
    - Open webcam
    - Detect and recognize faces
    - Mark attendance for recognized students
    - Each attendance event is logged with name, ID, time, and date.
2.4 View Enrolled Students
    Displays a list of all students whose images are enrolled in the system.
    Useful to verify if data collection is complete.
2.5 View Attendance Logs
    Opens or displays the daily attendance logs stored in .csv or Excel-compatible format.
    Each day's attendance is saved as a separate file.

3️. Attendance Logs Format
Attendance logs are saved 

4. Attendance Logs in Excel
Student ID	Name	Date	Time In
1001	Anjali Sharma	2025-07-17	09:01:32 AM
1002	Rahul Mehta	2025-07-17	09:03:45 AM


