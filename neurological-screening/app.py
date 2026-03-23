from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
import tkinter
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.linalg import pinv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
import sklearn.preprocessing

_original_lb_init = sklearn.preprocessing.LabelBinarizer.__init__

def patched_lb_init(self, neg_label=-1, pos_label=1, sparse_output=False):
    self.neg_label = neg_label
    self.pos_label = pos_label
    self.sparse_output = sparse_output

sklearn.preprocessing.LabelBinarizer.__init__ = patched_lb_init

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier

from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM, Bidirectional

import scipy.linalg
if not hasattr(scipy.linalg, "pinv2"):
    scipy.linalg.pinv2 = scipy.linalg.pinv

# Fix LabelBinarizer issue for sklearn_extensions compatibility
import sklearn.preprocessing
sklearn.preprocessing.LabelBinarizer.__init__ = (
    lambda self, neg_label=-1, pos_label=1, sparse_output=False:
    setattr(self, 'neg_label', neg_label) or
    setattr(self, 'pos_label', pos_label) or
    setattr(self, 'sparse_output', sparse_output)
)

# ======================= Import AtaxiaFusionNet =======================
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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


def ataxia_predict(test_path, model_path="outputs/ataxia_fusion_model.pth"):
    df = pd.read_csv(test_path)
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_seq_tensor.to(device), X_kin_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    df_original["Predicted_Label"] = ["ataxia" if p == 1 else "normal" for p in preds]
    df_original["Confidence"] = probs

    output_path = "outputs/final_predictions.csv"
    os.makedirs("outputs", exist_ok=True)
    df_original.to_csv(output_path, index=False)
    return df_original, output_path

# ======================= Tkinter Setup =======================

main = tkinter.Tk()
main.title("Automatic Detection of Genetic Diseases in Pediatric Age Using Pupillometry")
main.geometry("1300x1200")
main.config(bg='turquoise')

# ======================= Global Variables =======================
filename = None
classifier = None
left_X_train = left_X_test = left_y_train = left_y_test = None
right_X_train = right_X_test = right_y_train = right_y_test = None
left_X = left_Y = None
left_pupil = right_pupil = None
count = left = right = ids = None
left_svm_acc = right_svm_acc = 0
ensemble_acc = elm_acc = lstm_acc = bilstm_acc = 0

# ======================= Functions =======================
def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, 'Pupillometric dataset loaded\n')

def filtering():
    global left_pupil, right_pupil, count, left, right, ids
    left_pupil, right_pupil = [], []
    count, ids = 0, 1
    left = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    right = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'

    for root, dirs, directory in os.walk('dataset'):
        for file in directory:
            file_path = os.path.join('dataset', file)
            with open(file_path, 'r') as filedata:
                lines = filedata.readlines()
                left_pupil.clear()
                right_pupil.clear()
                count = 0
                for line in lines:
                    arr = line.strip().split("\t")
                    if len(arr) == 8 and arr[7] == '.....':
                        left_pupil.append(float(arr[3].strip()))
                        right_pupil.append(float(arr[6].strip()))
                        count += 1
                        if count == 100:
                            left_min, right_min = min(left_pupil), min(right_pupil)
                            left_max, right_max = max(left_pupil), max(right_pupil)
                            left_delta, right_delta = left_max - left_min, right_max - right_min
                            left_CH, right_CH = left_delta / left_max, right_delta / right_max
                            latency = 0.5
                            left_MCV = left_delta / (left_min - latency)
                            right_MCV = right_delta / (right_min - latency)
                            count = 0
                            left_pupil.clear()
                            right_pupil.clear()

                            left_label = 1 if left_min > 500 and left_max > 500 else 0
                            right_label = 1 if right_min > 500 and right_max > 500 else 0

                            left += f"{ids},{left_max},{left_min},{left_delta},{left_CH},{latency},{left_MCV},{left_label}\n"
                            right += f"{ids},{right_max},{right_min},{right_delta},{right_CH},{latency},{right_MCV},{right_label}\n"
                            ids += 1

    text.delete('1.0', END)
    text.insert(END, 'Feature filtration process completed\n')
    text.insert(END, f'Total patients found in dataset: {ids}\n')

def featuresExtraction():
    with open("left.txt", "w") as f:
        f.write(left)
    with open("right.txt", "w") as f:
        f.write(right)
    text.delete('1.0', END)
    text.insert(END, 'Features saved in left.txt and right.txt\n')

def featuresReduction():
    global left_X, left_Y, left_X_train, left_X_test, left_y_train, left_y_test
    global right_X_train, right_X_test, right_y_train, right_y_test

    left_pupil_df = pd.read_csv('left.txt')
    right_pupil_df = pd.read_csv('right.txt')

    left_X = left_pupil_df.iloc[:, 1:-1].values
    left_Y = left_pupil_df.iloc[:, -1].values
    right_X = right_pupil_df.iloc[:, 1:-1].values
    right_Y = right_pupil_df.iloc[:, -1].values

    left_X = normalize(left_X)
    right_X = normalize(right_X)

    left_X_train, left_X_test, left_y_train, left_y_test = train_test_split(left_X, left_Y, test_size=0.2, random_state=42)
    right_X_train, right_X_test, right_y_train, right_y_test = train_test_split(right_X, right_Y, test_size=0.2, random_state=42)

    text.delete('1.0', END)
    text.insert(END, f"Left pupil training: {len(left_X_train)}, testing: {len(left_X_test)}\n")
    text.insert(END, f"Right pupil training: {len(right_X_train)}, testing: {len(right_X_test)}\n")

def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    return y_pred

def rightSVM():
    global right_classifier, right_svm_acc
    right_classifier = svm.SVC(kernel='rbf', probability=True)
    right_classifier.fit(right_X_train, right_y_train)
    preds = prediction(right_X_test, right_classifier)
    acc = accuracy_score(right_y_test, preds) * 100
    right_svm_acc = acc
    text.delete('1.0', END)
    text.insert(END, f"Right Eye SVM Accuracy: {acc:.2f}%\n")

def leftSVM():
    global left_classifier, left_svm_acc
    left_classifier = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
    left_classifier.fit(left_X_train, left_y_train)
    preds = prediction(left_X_test, left_classifier)
    acc = accuracy_score(left_y_test, preds) * 100
    left_svm_acc = acc
    text.delete('1.0', END)
    text.insert(END, f"Left Eye SVM Accuracy: {acc:.2f}%\n")

def ensemble():
    global classifier, ensemble_acc
    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))
    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    l_svm = svm.SVC(kernel='linear', probability=True)
    r_svm = svm.SVC(kernel='linear', probability=True)
    classifier = VotingClassifier(estimators=[('Left', l_svm), ('Right', r_svm)], voting='hard')
    classifier.fit(trainX, trainY)
    preds = prediction(testX, classifier)
    acc = accuracy_score(testY, preds) * 100
    ensemble_acc = acc
    text.delete('1.0', END)
    text.insert(END, f"Ensemble Accuracy: {acc:.2f}%\n")



def extension():
    global elm_acc
    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))
    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    srhl_tanh = MLPRandomLayer(n_hidden=100, activation_func='tanh')
    elm = GenELMClassifier(hidden_layer=srhl_tanh)
    elm.fit(trainX, trainY)
    preds = prediction(testX, elm)
    elm_acc = accuracy_score(testY, preds) * 100
    text.delete('1.0', END)
    text.insert(END, f"ELM Accuracy: {elm_acc:.2f}%\n")

def runLSTM():
    global lstm_acc
    Y = left_Y.reshape(-1, 1)
    enc = OneHotEncoder(sparse=False)
    Y = enc.fit_transform(Y)
    X = left_X.reshape((left_X.shape[0], left_X.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], 1)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X, Y, epochs=10, batch_size=5, verbose=2)
    lstm_acc = hist.history['accuracy'][-1] * 100
    text.insert(END, f"LSTM Accuracy: {lstm_acc:.2f}%\n")

def runBILSTM():
    global bilstm_acc
    Y = left_Y.reshape(-1, 1)
    enc = OneHotEncoder(sparse=False)
    Y = enc.fit_transform(Y)
    X = left_X.reshape((left_X.shape[0], left_X.shape[1], 1))

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1))),
        Bidirectional(LSTM(32, return_sequences=True)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X, Y, epochs=10, batch_size=5, verbose=2)
    bilstm_acc = hist.history['accuracy'][-1] * 100
    text.insert(END, f"BiLSTM Accuracy: {bilstm_acc:.2f}%\n")

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testData")
    test = pd.read_csv(filename).values[:, :7]
    text.insert(END, f"{filename} loaded for testing\n")
    preds = classifier.predict(test)
    for i, p in enumerate(preds):
        label = "Disease detected" if p == 1 else "No disease detected"
        text.insert(END, f"X={test[i]}, Predicted={label}\n")

def runAtaxiaFusionNet():
    text.delete('1.0', END)
    test_file = filedialog.askopenfilename(title="Select test CSV for AtaxiaFusionNet")
    if not test_file:
        return
    text.insert(END, f"Running AtaxiaFusionNet on: {test_file}\n")
    try:
        df_result, output_path = ataxia_predict(test_file)
        text.insert(END, f"Predictions complete. Saved to {output_path}\n\n")
        text.insert(END, df_result.head(20).to_string(index=False))
    except Exception as e:
        messagebox.showerror("Error", f"Error running AtaxiaFusionNet:\n{str(e)}")

def graph():
    accs = [right_svm_acc, left_svm_acc, ensemble_acc, elm_acc, lstm_acc, bilstm_acc]
    bars = ['Right SVM', 'Left SVM', 'Ensemble', 'ELM', 'LSTM', 'BiLSTM']
    plt.bar(bars, accs)
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Comparison')
    plt.show()

# ======================= GUI Layout =======================
font = ('times', 16, 'bold')
Label(main, text='Automatic Detection of Genetic Diseases in Pediatric Age Using Pupillometry',
      bg='dark goldenrod', fg='white', font=font, height=3, width=120).place(x=0, y=5)

font1 = ('times', 13, 'bold')
Button(main, text="Upload Pupillometric Dataset", command=upload, font=font1).place(x=700, y=100)
pathlabel = Label(main, bg='DarkOrange1', fg='white', font=font1)
pathlabel.place(x=700, y=150)

buttons = [
    ("Run Filtering", filtering),
    ("Run Features Extraction", featuresExtraction),
    ("Run Features Reduction", featuresReduction),
    ("Run SVM on Right Eye Features", rightSVM),
    ("Run SVM on Left Eye Features", leftSVM),
    ("Run OR Ensemble Algorithm (Left & Right SVM)", ensemble),
    ("Run Extension Extreme Learning Machine Algorithm", extension),
    ("Run LSTM", runLSTM),
    ("Run BiLSTM", runBILSTM),
    ("Run AtaxiaFusionNet Prediction", runAtaxiaFusionNet),
    ("Accuracy Graph with Metrics", graph),
    ("Predict Disease", predict)
]

y = 200
for label, cmd in buttons:
    Button(main, text=label, command=cmd, font=font1).place(x=700, y=y)
    y += 50

text = Text(main, height=30, width=80, font=('times', 12, 'bold'))
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)

main.mainloop()
