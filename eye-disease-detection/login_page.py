import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import sys
import os

# ================================================
# Cross-Modal Disease Prediction - Login Interface
# ================================================

class CrossModalLogin:
    def __init__(self, root):
        self.root = root
        self.root.title("Cross-Modal Disease Prediction Login")
        self.root.geometry("950x550")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f7fa")

        # -------------------------------
        # Left Panel (Title + Illustration)
        # -------------------------------
        left_frame = tk.Frame(self.root, bg="#004c6d", width=380, height=550)
        left_frame.pack(side="left", fill="y")

        title_label = tk.Label(
            left_frame,
            text="Cross-Modal Disease Prediction\nUsing Gait and Pupillary Biometrics",
            font=("Segoe UI Semibold", 16),
            bg="#004c6d",
            fg="white",
            wraplength=340,
            justify="center"
        )
        title_label.place(x=20, y=140)

        subtitle_label = tk.Label(
            left_frame,
            text="For Pediatric Neurological & Genetic Disorders",
            font=("Segoe UI", 11, "italic"),
            bg="#004c6d",
            fg="#d0e8f2"
        )
        subtitle_label.place(x=35, y=230)

        try:
            img = Image.open("biometric_bg.jpg")
            img = img.resize((340, 240))
            self.bg_img = ImageTk.PhotoImage(img)
            img_label = tk.Label(left_frame, image=self.bg_img, bg="#004c6d")
            img_label.place(x=20, y=280)
        except:
            pass  # Ignore if image not found

        # -------------------------------
        # Right Panel (Login Fields)
        # -------------------------------
        right_frame = tk.Frame(self.root, bg="#f5f7fa", width=570, height=550)
        right_frame.pack(side="right", fill="y")

        header = tk.Label(
            right_frame,
            text="User Login Portal",
            font=("Segoe UI Bold", 22),
            bg="#f5f7fa",
            fg="#004c6d"
        )
        header.place(x=170, y=100)

        tk.Label(
            right_frame,
            text="Username",
            font=("Segoe UI", 12),
            bg="#f5f7fa",
            fg="#333333"
        ).place(x=150, y=200)
        self.username_entry = tk.Entry(right_frame, font=("Segoe UI", 12), width=30, bd=0, highlightthickness=2)
        self.username_entry.place(x=150, y=230)
        self.username_entry.config(highlightbackground="#cccccc", highlightcolor="#004c6d")

        tk.Label(
            right_frame,
            text="Password",
            font=("Segoe UI", 12),
            bg="#f5f7fa",
            fg="#333333"
        ).place(x=150, y=280)
        self.password_entry = tk.Entry(right_frame, font=("Segoe UI", 12), width=30, bd=0, show="*", highlightthickness=2)
        self.password_entry.place(x=150, y=310)
        self.password_entry.config(highlightbackground="#cccccc", highlightcolor="#004c6d")

        login_btn = tk.Button(
            right_frame,
            text="Login",
            command=self.login,
            font=("Segoe UI Semibold", 13),
            bg="#004c6d",
            fg="white",
            activebackground="#005f88",
            activeforeground="white",
            cursor="hand2",
            bd=0,
            relief="flat",
            width=18,
            height=1
        )
        login_btn.place(x=190, y=380)

        # tk.Label(
        #     right_frame,
        #     text="Forgot Password?",
        #     font=("Segoe UI", 10, "underline"),
        #     bg="#f5f7fa",
        #     fg="#004c6d",
        #     cursor="hand2"
        # ).place(x=240, y=430)

        tk.Label(
            right_frame,
            text="© 2025 Biomedical AI Research Group",
            font=("Segoe UI", 9),
            bg="#f5f7fa",
            fg="#999999"
        ).place(x=180, y=510)

    # =================================================
    # Authentication Logic
    # =================================================
    def login(self):
        user = self.username_entry.get().strip()
        pwd = self.password_entry.get().strip()

        if not user or not pwd:
            messagebox.showwarning("Missing Fields", "Please enter both username and password.")
            return

        if user == "admin" and pwd == "admin":
            messagebox.showinfo("Access Granted", f"Welcome, {user}.")
            self.root.destroy()
            self.open_homepage()
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    def open_homepage(self):
        """Open homepage.py after successful login"""
        try:
            homepage_script = os.path.join(os.getcwd(), "homepage.py")
            if not os.path.exists(homepage_script):
                raise FileNotFoundError("homepage.py not found in current directory.")
            subprocess.Popen([sys.executable, homepage_script])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Unable to open homepage:\n{e}")


# ============================
# Main Entry Point
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = CrossModalLogin(root)
    root.mainloop()
