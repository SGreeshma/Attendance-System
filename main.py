from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import os
import pandas as pd

import openpyxl
from capture_images import capture_images
from train_model import train_model
from recognize_attendance import recognize_with_blink

CREDENTIALS = {
    "admin": "admin123"
}

COLORS = {
    "background": "#f0f2f5",
    "card_bg": "#ffffff",
    "primary": "#3f51b5",
    "text": "#2e3b4e",
    "text_light": "#757575",
    "success": "#4caf50",
    "danger": "#e53935",
    "border": "#e3f2fd",
    "border_focus": "#90caf9"
}

class ModernEntry(tk.Entry):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            font=("Segoe UI", 12),
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border_focus"]
        )
        self.bind("<FocusIn>", self.on_focus_in)
        self.bind("<FocusOut>", self.on_focus_out)

    def on_focus_in(self, e):
        self.configure(highlightbackground=COLORS["border_focus"])

    def on_focus_out(self, e):
        self.configure(highlightbackground=COLORS["border"])

class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(font=("Segoe UI", 12), bg=COLORS["primary"], fg="white")

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📷 Face Recognition Attendance System")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f2f5")
        self.default_font = ("Segoe UI", 12)
        self.show_login_screen()

    def show_login_screen(self):
        self.clear_screen()

        # Center container with gradient background
        center_container = tk.Frame(self.root, bg=COLORS["background"])
        center_container.pack(expand=True, fill="both")

        # Main container with shadow effect
        main_frame = tk.Frame(
            center_container,
            bg=COLORS["card_bg"],
            padx=40,
            pady=40
        )
        main_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Content container
        content_frame = tk.Frame(
            main_frame,
            bg=COLORS["card_bg"],
            width=500  # Increased width
        )
        content_frame.pack(fill="both", expand=True)

        # Logo and Title with modern styling
        logo_label = tk.Label(
            content_frame,
            text="👨‍🎓",
            font=("Segoe UI", 72),
            bg=COLORS["card_bg"],
            fg=COLORS["primary"]
        )
        logo_label.pack(pady=(0, 20))

        title_label = tk.Label(
            content_frame,
            text="Student Attendance",
            font=("Segoe UI", 32, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            justify="center"
        )
        title_label.pack(pady=(0, 10))

        subtitle_label = tk.Label(
            content_frame,
            text="Faculty Portal",
            font=("Segoe UI", 16),
            bg=COLORS["card_bg"],
            fg=COLORS["text_light"]
        )
        subtitle_label.pack(pady=(0, 30))

        # Form container with modern styling
        form_frame = tk.Frame(
            content_frame,
            bg=COLORS["card_bg"],
            padx=30,
            pady=30
        )
        form_frame.pack(fill="x")

        # Username field with icon and complete styling
        username_frame = tk.Frame(form_frame, bg=COLORS["card_bg"])
        username_frame.pack(fill="x", pady=(0, 20))
        
        username_label = tk.Label(
            username_frame,
            text="👤 Username",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            anchor="w"
        )
        username_label.pack(fill="x", pady=(0, 8))
        
        self.username_entry = ModernEntry(username_frame)
        self.username_entry.configure(
            font=("Segoe UI", 14),
            width=40
        )
        self.username_entry.pack(fill="x", ipady=10)

        # Password field with icon and complete styling
        password_frame = tk.Frame(form_frame, bg=COLORS["card_bg"])
        password_frame.pack(fill="x", pady=(0, 30))
        
        password_label = tk.Label(
            password_frame,
            text="🔒 Password",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            anchor="w"
        )
        password_label.pack(fill="x", pady=(0, 8))
        
        self.password_entry = ModernEntry(password_frame, show="•")
        self.password_entry.configure(
            font=("Segoe UI", 14),
            width=40
        )
        self.password_entry.pack(fill="x", ipady=10)

        # Login button with modern styling
        login_btn = ModernButton(
            form_frame,
            text="Login to Dashboard",
            command=self.login,
            width=30
        )
        login_btn.configure(
            font=("Segoe UI", 14, "bold"),
            pady=12
        )
        login_btn.pack(fill="x", pady=(10, 0))

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if CREDENTIALS.get(username) == password:
            self.show_dashboard()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials")

    def show_dashboard(self):
        self.clear_screen()

        # Main container with modern styling
        main_frame = tk.Frame(self.root, bg=COLORS["background"], padx=40, pady=40)
        main_frame.pack(expand=True, fill="both")

        # Header with modern design
        header_frame = tk.Frame(main_frame, bg=COLORS["card_bg"], padx=30, pady=25)
        header_frame.pack(fill="x", pady=(0, 30))

        # Welcome section with modern layout
        welcome_frame = tk.Frame(header_frame, bg=COLORS["card_bg"])
        welcome_frame.pack(side="left")

        logo_label = tk.Label(
            welcome_frame,
            text="👨‍🏫",
            font=("Segoe UI", 36),
            bg=COLORS["card_bg"],
            fg=COLORS["primary"]
        )
        logo_label.pack(side="left", padx=(0, 15))

        welcome_text_frame = tk.Frame(welcome_frame, bg=COLORS["card_bg"])
        welcome_text_frame.pack(side="left")

        welcome_label = tk.Label(
            welcome_text_frame,
            text="Welcome Back,",
            font=("Segoe UI", 16),
            bg=COLORS["card_bg"],
            fg=COLORS["text_light"]
        )
        welcome_label.pack(anchor="w")

        faculty_label = tk.Label(
            welcome_text_frame,
            text="Faculty",
            font=("Segoe UI", 28, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        )
        faculty_label.pack(anchor="w")

        # Logout button with modern styling
        logout_btn = ModernButton(
            header_frame,
            text="🚪 Logout",
            command=self.show_login_screen,
            bg=COLORS["danger"]
        )
        logout_btn.pack(side="right")

        # Dashboard buttons container with modern vertical layout
        buttons_frame = tk.Frame(main_frame, bg=COLORS["card_bg"], padx=30, pady=30)
        buttons_frame.pack(expand=True, fill="both")

        # Create a vertical list of buttons
        button_data = [
            ("📸 Capture Student Images", self.capture_images_gui),
            ("🧠 Train Model", self.train_model_gui),
            ("🕵️‍♂️ Start Attendance", self.start_recognition_gui),
            ("📋 View Enrolled Students", self.show_enrolled_students),
            ("📑 View Attendance Logs", self.show_attendance_logs)
        ]

        # Store button references
        self.buttons = {}
        for text, command in button_data:
            btn = ModernButton(
                buttons_frame,
                text=text,
                command=command
            )
            btn.pack(fill="x", pady=10)
            # Store button reference with a key based on its text
            if "Train Model" in text:
                self.train_btn = btn
            elif "Start Attendance" in text:
                self.recognition_btn = btn

        # Status label with modern styling
        self.status_label = tk.Label(
            main_frame,
            text="Status: Ready",
            font=("Segoe UI", 12),
            fg=COLORS["success"],
            bg=COLORS["background"],
            pady=20
        )
        self.status_label.pack(pady=20)

        self.update_button_states()

    def capture_images_gui(self):
        from tkinter.simpledialog import askstring

        name = askstring("Student Info", "Enter student name:")
        roll = askstring("Student Info", "Enter student roll number:")

        if not name or not roll:
            messagebox.showwarning("Missing Info", "Both name and roll number are required.")
            return

        self.status_label.config(text="Status: Capturing images...", fg="blue")
        try:
            count = capture_images(name.strip(), roll.strip())
            self.status_label.config(text=f"✅ Captured {count} images for {roll.strip()}", fg="green")
        except Exception as e:
            messagebox.showerror("Capture Error", str(e))
            self.status_label.config(text="❌ Failed to capture images", fg="red")

        self.update_button_states()

    def train_model_gui(self):
        self.status_label.config(text="Status: Training model...", fg="blue")
        try:
            train_model()
            self.status_label.config(text="✅ Model trained successfully", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="❌ Training failed", fg="red")
        self.update_button_states()

    def start_recognition_gui(self):
        self.status_label.config(text="Status: Starting recognition...", fg="blue")
        try:
            recognize_with_blink()
            self.status_label.config(text="✅ Recognition completed", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="❌ Recognition failed", fg="red")
        self.update_button_states()
    
    def show_attendance_logs(self):
        self.clear_screen()

        frame = tk.Frame(self.root, bg="#f0f2f5", pady=20)
        frame.pack(expand=True)

        tk.Label(frame, text="Attendance Logs", font=("Segoe UI", 24, "bold"), bg="#f0f2f5", fg="#2e3b4e").pack(pady=20)

        log_display = tk.Text(frame, font=self.default_font, width=60, height=25, wrap="word", padx=10, pady=10)
        log_display.pack(pady=20)

        records_dir = "attendanceRecords"
        logs = []

        try:
            if os.path.exists(records_dir):
                for filename in sorted(os.listdir(records_dir)):
                    if filename.endswith(".xlsx"):
                        date_part = filename.replace(".xlsx", "")
                        file_path = os.path.join(records_dir, filename)
                        df = pd.read_excel(file_path)

                        for _, row in df.iterrows():
                            roll_number = row.get("Roll Number")
                            time_str = row.get("Time")
                            logs.append(f"{date_part} {time_str} - Roll: {roll_number}")
            else:
                log_display.insert(tk.END, "No attendance records found.")

            if logs:
                logs.sort()  # Optional: to sort logs chronologically
                log_display.insert(tk.END, "\n".join(logs))
            else:
                log_display.insert(tk.END, "No attendance logs available yet.")
        except Exception as e:
            log_display.insert(tk.END, f"Error reading logs: {str(e)}")

        tk.Button(frame, text="Back to Dashboard", font=self.default_font, bg="#3f51b5", fg="white", width=20, pady=5,
              command=self.show_dashboard).pack(pady=20)

    def show_enrolled_students(self):
        pics_path = "pics"
        if not os.path.exists(pics_path):
            messagebox.showinfo("Student List", "No students enrolled yet.")
            return

        students = []
        for folder in os.listdir(pics_path):
            folder_path = os.path.join(pics_path, folder)
            if os.path.isdir(folder_path):
                parts = folder.split("_", 1)
                if len(parts) == 2:
                    roll, name = parts
                    students.append(f"{roll} - {name}")
                else:
                    students.append(folder)

        if students:
            student_list = "\n".join(sorted(students))
            messagebox.showinfo("Enrolled Students", student_list)
        else:
            messagebox.showinfo("Enrolled Students", "No students enrolled yet.")

    def update_button_states(self):
        has_images = os.path.exists("pics") and any(
            os.listdir(os.path.join("pics", d)) for d in os.listdir("pics")
            if os.path.isdir(os.path.join("pics", d))
        )
        has_model = os.path.exists("trainer/trainer.yml") and os.path.exists("trainer/labels.pkl")

        if hasattr(self, 'train_btn'):
            self.train_btn.configure(state="normal" if has_images else "disabled")
        if hasattr(self, 'recognition_btn'):
            self.recognition_btn.configure(state="normal" if has_model else "disabled")

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
