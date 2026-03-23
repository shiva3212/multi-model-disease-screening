import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ===================================================
# MODEL DEFINITION
# ===================================================
class AtaxiaFusionNet(nn.Module):
    def __init__(self, input_dim_seq, input_dim_kin, lstm_hidden=64, mlp_hidden=64, fusion_hidden=64):
        super(AtaxiaFusionNet, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_dim_seq, hidden_size=lstm_hidden,
                              batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim_kin, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * lstm_hidden + mlp_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, 2)
        )

    def forward(self, x_seq, x_kin):
        _, (hn, _) = self.bilstm(x_seq)
        lstm_out = torch.cat((hn[-2], hn[-1]), dim=1)
        mlp_out = self.mlp(x_kin)
        combined = torch.cat((lstm_out, mlp_out), dim=1)
        return self.fusion(combined)


# ===================================================
# ATAXIA INFORMATION
# ===================================================
ataxia_info = {
    "disease": "Cerebellar Ataxia",
    "cause": "Cerebellum damage",
    "symptoms": "Unsteady walk, tremor, slurred speech",
    "prevention": "Avoid alcohol, ensure vitamins B1 B12 E",
    "treatment": "Vitamin supplements, physiotherapy, medical support"
}


# ===================================================
# GUI APPLICATION
# ===================================================
class AtaxiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neurological Disorder (Ataxia) Detection Dashboard")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f9fafc")

        self.model_path = "outputs/ataxia_fusion_model.pth"
        self.text_area = scrolledtext.ScrolledText(root, width=95, height=25, font=("Consolas", 10))
        self.text_area.place(x=30, y=230)

        header = tk.Label(root, text="Cerebellar Ataxia Detection Dashboard",
                          font=("Segoe UI Bold", 22), bg="#f9fafc", fg="#003049")
        header.place(x=250, y=30)

        # Buttons
        tk.Button(root, text="Browse Test Data", command=self.browse_csv,
                  bg="#004c6d", fg="white", font=("Segoe UI", 11), width=20).place(x=150, y=120)
        tk.Button(root, text="Predict", command=self.predict_data,
                  bg="#006d77", fg="white", font=("Segoe UI", 11), width=20).place(x=400, y=120)
        tk.Button(root, text="Show Graphs", command=self.show_graphs,
                  bg="#5a189a", fg="white", font=("Segoe UI", 11), width=20).place(x=650, y=120)

        self.test_path = None

    # ===================================================
    # BROWSE FILE
    # ===================================================
    def browse_csv(self):
        path = filedialog.askopenfilename(title="Select Test CSV File",
                                          filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        self.test_path = path
        self.text_area.insert(tk.END, f"\nLoaded test data: {path}\n")

    # ===================================================
    # PREDICT FUNCTION
    # ===================================================
    def predict_data(self):
        if not self.test_path:
            messagebox.showwarning("No File", "Please select a CSV file first.")
            return

        df = pd.read_csv(self.test_path)
        df_original = df.copy()

        if "label" in df.columns:
            X = df.drop(columns=["label"])
        else:
            X = df.copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)

        seq_len = 5
        pad_width = seq_len - (X_scaled.shape[1] % seq_len) if X_scaled.shape[1] % seq_len != 0 else 0
        if pad_width > 0:
            X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode="constant")

        segment_dim = X_scaled.shape[1] // seq_len
        X_seq = X_scaled.reshape(-1, seq_len, segment_dim)
        X_kin = X_scaled

        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
        X_kin_tensor = torch.tensor(X_kin, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AtaxiaFusionNet(input_dim_seq=segment_dim, input_dim_kin=X_kin.shape[1])
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(X_seq_tensor.to(device), X_kin_tensor.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

        df_original["Predicted_Label"] = ["ataxia" if p == 1 else "normal" for p in preds]
        df_original["Confidence"] = probs

        self.text_area.insert(tk.END, "\n=== Prediction Results (First 10 Rows) ===\n")
        self.text_area.insert(tk.END, df_original.head(10).to_string(index=False))
        self.text_area.insert(tk.END, f"\n\nTotal Records: {len(df_original)}")
        self.text_area.insert(tk.END, f"\nPredicted Ataxia: {sum(preds)} | Normal: {len(preds) - sum(preds)}\n")

        # --- Recommendation Section ---
        if sum(preds) > 0:
            self.text_area.insert(tk.END, "\n--- Medical Recommendation ---\n")
            self.text_area.insert(tk.END, f"Disease: {ataxia_info['disease']}\n")
            self.text_area.insert(tk.END, f"Cause: {ataxia_info['cause']}\n")
            self.text_area.insert(tk.END, f"Symptoms: {ataxia_info['symptoms']}\n")
            self.text_area.insert(tk.END, f"Prevention: {ataxia_info['prevention']}\n")
            self.text_area.insert(tk.END, f"Treatment: {ataxia_info['treatment']}\n\n")
        else:
            self.text_area.insert(tk.END, "\nAll subjects are normal. No medical recommendation needed.\n\n")

        # Save
        os.makedirs("outputs", exist_ok=True)
        df_original.to_csv("outputs/final_predictions.csv", index=False)
        self.text_area.insert(tk.END, "\nPredictions saved to outputs/final_predictions.csv\n")

    # ===================================================
    # SHOW GRAPHS
    # ===================================================
    def show_graphs(self):
        plots = [
            "Figure_1 - Copy.png", "Figure_2 - Copy.png",
            "plot1_class_distribution.png", "plot3_boxplot.png",
            "plot4_pairplot.png", "plot5_distribution.png"
        ]
        for img_file in plots:
            if os.path.exists(img_file):
                img = plt.imread(img_file)
                plt.figure(figsize=(6, 5))
                plt.imshow(img)
                plt.axis("off")
                plt.title(os.path.basename(img_file))
                plt.show()
            else:
                self.text_area.insert(tk.END, f"\nFile not found: {img_file}\n")


# ===================================================
# MAIN
# ===================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AtaxiaApp(root)
    root.mainloop()
