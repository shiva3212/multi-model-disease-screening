import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------- Model Definition -------------------
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

# ------------------- Prediction Function -------------------
def predict(test_path, model_path="outputs/ataxia_fusion_model.pth"):
    print(f"\nLoading test data from: {test_path}")

    # Load test data
    df = pd.read_csv(test_path)
    df_original = df.copy()  # keep original data for display and saving

    # Drop label if present
    if "label" in df.columns:
        X = df.drop(columns=["label"])
    else:
        X = df.copy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Prepare sequence format
    seq_len = 5
    pad_width = seq_len - (X_scaled.shape[1] % seq_len) if X_scaled.shape[1] % seq_len != 0 else 0
    if pad_width > 0:
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode="constant")

    segment_dim = X_scaled.shape[1] // seq_len
    X_seq = X_scaled.reshape(-1, seq_len, segment_dim)
    X_kin = X_scaled

    # Tensors
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
    X_kin_tensor = torch.tensor(X_kin, dtype=torch.float32)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AtaxiaFusionNet(input_dim_seq=segment_dim, input_dim_kin=X_kin.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(X_seq_tensor.to(device), X_kin_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    # Append predictions to dataframe
    df_original["Predicted_Label"] = ["ataxia" if p == 1 else "normal" for p in preds]
    df_original["Confidence"] = probs

    # Display first few rows
    print("\n=== Predictions Preview (first 20 records) ===")
    print(df_original.head(20).to_string(index=False))
    print(f"\nTotal Records: {len(df_original)}")
    print(f"Predicted Ataxia: {sum(preds)} | Normal: {len(preds) - sum(preds)}")

    # Save final output
    output_path = "outputs/final_predictions.csv"
    df_original.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

# ------------------- Run Prediction -------------------
if __name__ == "__main__":
    # Specify your test file path here
    test_csv_path = "outputs/test_data.csv"   # Change to your test file path
    predict(test_csv_path)
