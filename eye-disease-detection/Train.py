import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------- Data Preparation -------------------
df = pd.read_csv("balanced_ataxia_dataset.csv")
non_feature_cols = ['label', 'source_file'] if 'source_file' in df.columns else ['label']
X = df.drop(columns=non_feature_cols)
y = df['label'].map({'ataxia': 1, 'normal': 0})

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure features are divisible by seq_len
seq_len = 5
pad_width = seq_len - (X_scaled.shape[1] % seq_len) if X_scaled.shape[1] % seq_len != 0 else 0
if pad_width > 0:
    X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')

segment_dim = X_scaled.shape[1] // seq_len

# Prepare sequence input
X_seq = X_scaled.reshape(-1, seq_len, segment_dim)
X_kin = X_scaled

# Split
X_seq_train, X_seq_test, X_kin_train, X_kin_test, y_train, y_test = train_test_split(
    X_seq, X_kin, y, stratify=y, test_size=0.2, random_state=42
)

# ------------------- Dataset Class -------------------
class AtaxiaDataset(Dataset):
    def __init__(self, X_seq, X_kin, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_kin = torch.tensor(X_kin, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_kin[idx], self.y[idx]

train_loader = DataLoader(AtaxiaDataset(X_seq_train, X_kin_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(AtaxiaDataset(X_seq_test, X_kin_test, y_test), batch_size=16)

# ------------------- Hybrid Model -------------------
class AtaxiaFusionNet(nn.Module):
    def __init__(self, input_dim_seq, input_dim_kin, lstm_hidden=64, mlp_hidden=64, fusion_hidden=64):
        super(AtaxiaFusionNet, self).__init__()
        # BiLSTM for spatio-temporal features
        self.bilstm = nn.LSTM(input_size=input_dim_seq, hidden_size=lstm_hidden,
                              batch_first=True, bidirectional=True)
        # MLP for kinematic features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim_kin, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        # Fusion layer
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

# ------------------- Training Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AtaxiaFusionNet(input_dim_seq=segment_dim, input_dim_kin=X_kin.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# ------------------- Train and Evaluate -------------------
def train(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x_seq, x_kin, y in loader:
        x_seq, x_kin, y = x_seq.to(device), x_kin.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_seq, x_kin)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for x_seq, x_kin, y in loader:
            x_seq, x_kin, y = x_seq.to(device), x_kin.to(device), y.to(device)
            outputs = model(x_seq, x_kin)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return total_loss / total, correct / total, y_true, y_pred, y_proba

# ------------------- Training Loop -------------------
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc, _, _, _ = evaluate(model, test_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} | Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# ------------------- Final Evaluation -------------------
_, _, y_true, y_pred, y_proba = evaluate(model, test_loader)
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_true, y_proba))

# ------------------- Save Model and Test Data -------------------
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/ataxia_fusion_model.pth")

test_df = pd.DataFrame(X_kin_test, columns=[f"feature_{i}" for i in range(X_kin_test.shape[1])])
test_df["label"] = y_test.values
test_df.to_csv("outputs/test_data.csv", index=False)

print("\nModel and test data saved successfully:")
print("→ Model file: outputs/ataxia_fusion_model.pth")
print("→ Test data: outputs/test_data.csv")

# ------------------- Visualization -------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Ataxia'], yticklabels=['Normal', 'Ataxia'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
