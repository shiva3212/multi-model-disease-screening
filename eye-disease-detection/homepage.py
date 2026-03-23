import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys

# ============================================================
# Home Page: Cross-Modal Disease Prediction System
# ============================================================

class CrossModalHome:
    def __init__(self, root):
        self.root = root
        self.root.title("Cross-Modal Disease Prediction System")
        self.root.geometry("1000x600")
        self.root.config(bg="#f0f4f8")
        self.root.resizable(False, False)

        # ---------------- Header ----------------
        title = tk.Label(
            root,
            text="Cross-Modal Disease Prediction System",
            font=("Segoe UI Semibold", 24),
            bg="#004c6d",
            fg="white",
            pady=15,
            width=100
        )
        title.pack()

        subtitle = tk.Label(
            root,
            text="Integrating Gait and Pupillary Biometrics for Pediatric Neurological and Genetic Disorders",
            font=("Segoe UI", 13, "italic"),
            bg="#004c6d",
            fg="#d0e8f2",
            pady=5
        )
        subtitle.pack()

        # ---------------- Main Section ----------------
        frame = tk.Frame(root, bg="#f0f4f8", pady=40)
        frame.pack(expand=True)

        tk.Label(
            frame,
            text="Choose a Research Module",
            font=("Segoe UI Bold", 18),
            bg="#f0f4f8",
            fg="#00384e"
        ).pack(pady=20)

        # ---------------- Buttons ----------------
        disease_btn = tk.Button(
            frame,
            text="Pupillometry-Based Genetic Disease Detection",
            font=("Segoe UI", 14),
            bg="#006d91",
            fg="white",
            activebackground="#00516c",
            activeforeground="white",
            cursor="hand2",
            width=45,
            height=2,
            relief="flat",
            command=self.open_disease_detection
        )
        disease_btn.pack(pady=30)

        neuro_btn = tk.Button(
            frame,
            text="Neurological Disease Prediction (Ataxia FusionNet)",
            font=("Segoe UI", 14),
            bg="#004c6d",
            fg="white",
            activebackground="#00384e",
            activeforeground="white",
            cursor="hand2",
            width=45,
            height=2,
            relief="flat",
            command=self.open_neurology_module
        )
        neuro_btn.pack(pady=10)

        # ---------------- Footer ----------------
        footer = tk.Label(
            root,
            text="© 2025 Biomedical AI Research | Cross-Modal Gait & Pupillary Biometrics",
            font=("Segoe UI", 10),
            bg="#f0f4f8",
            fg="#888888"
        )
        footer.pack(side="bottom", pady=10)

    # ============================================================
    # Button Handlers
    # ============================================================

    def open_disease_detection(self):
        """Launch dd.py (Pupillometry-Based Genetic Disease Detection)"""
        try:
            script = os.path.join(os.getcwd(), "dd.py")
            if not os.path.exists(script):
                raise FileNotFoundError("dd.py not found in project directory.")
            subprocess.Popen([sys.executable, script])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to open Genetic Disease Detection module:\n{e}")

    def open_neurology_module(self):
        """Launch neurology.py (Neurological Disease Prediction GUI)"""
        try:
            script = os.path.join(os.getcwd(), "neurology.py")
            if not os.path.exists(script):
                raise FileNotFoundError("neurology.py not found in project directory.")
            subprocess.Popen([sys.executable, script])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to open Neurology module:\n{e}")


# ============================================================
# Main Entry
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CrossModalHome(root)
    root.mainloop()
