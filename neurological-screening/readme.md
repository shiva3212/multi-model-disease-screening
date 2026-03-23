# Cross-Modal Disease Prediction System: Codebase Explanation

This document provides a line-by-line explanation of the Python scripts and the PyTorch modeling analysis within the Cross-Modal Disease Prediction System. The application integrates a login portal, a main launcher (homepage), and two dashboard interfaces for neurological and genetic disease prediction, alongside a foundational Jupyter Notebook analyzing deep learning models for retinal disease classification.

---

## 1. `login page.py` - User Authentication Interface

This script sets up a graphical login screen using Tkinter, providing access control to the main system.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| **1-5** | `import tkinter as tk...` | Imports necessary libraries: `tkinter` for GUI, `messagebox` for alerts, `PIL`/`ImageTk` for image handling, and `subprocess`/`os`/`sys` for script execution. |
| **11-16** | `class CrossModalLogin: def __init__(self, root): self.root = root...` | Defines the main login class, setting up the Tkinter root window, title, fixed size (`950x550`), and disabling resizing. |
| **20-39** | `left_frame = tk.Frame(...) title_label = tk.Label(...)` | Creates the left, dark-themed **information panel**, including the application title and subtitle, and attempts to load a background image (`biometric_bg.jpg`). |
| **41-47** | `try: img = Image.open(...)` | **Image Handling:** Uses a `try-except` block to safely load and resize the background image using the PIL library. |
| **51-74** | `right_frame = tk.Frame(...) header = tk.Label(...) self.username_entry = tk.Entry(...)` | Creates the right, light-themed **login panel**, including the header, "Username" label, and the input field. |
| **75-77** | `self.password_entry = tk.Entry(..., show="*")` | Creates the "Password" input field, using `show="*"` to mask the characters. |
| **79-85** | `login_btn = tk.Button(..., command=self.login)` | Creates the **Login button**, linking it to the `self.login` method. |
| **93-96** | `def login(self): user = self.username_entry.get().strip()...` | **Authentication Logic:** Retrieves and cleans the username and password entered by the user. |
| **102-104** | `if user == "admin" and pwd == "admin": self.root.destroy(); self.open_homepage()` | **Credential Check:** If the hardcoded credentials ("admin"/"admin") are matched, the login window is closed, and the homepage is launched. |
| **108-113** | `def open_homepage(self): subprocess.Popen([sys.executable, homepage_script])` | **Homepage Launcher:** Safely launches `homepage.py` using the current Python interpreter as a separate process. |
| **118-121** | `if __name__ == "__main__": root = tk.Tk(); app = CrossModalLogin(root); root.mainloop()` | **Execution Block:** Standard Tkinter main loop initialization. |

---

## 2. `homepage.py` - Main Module Selector

This script serves as the main application hub, launched after a successful login, allowing the user to select a research module.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| **11-29** | `class CrossModalHome: def __init__(self, root): self.root = root...` | Defines the home class, setting up the main window and the blue-themed application title/subtitle header. |
| **39-49** | `disease_btn = tk.Button(..., command=self.open_disease_detection)` | Creates the button for the **"Pupillometry-Based Genetic Disease Detection"** module, linking it to its launcher method. |
| **51-61** | `neuro_btn = tk.Button(..., command=self.open_neurology_module)` | Creates the button for the **"Neurological Disease Prediction (Ataxia FusionNet)"** module, linking it to its launcher method. |
| **66-72** | `def open_disease_detection(self): script = os.path.join(..., "dd.py")... subprocess.Popen(...)` | **`dd.py` Launcher:** Defines the method to launch the `dd.py` script as a new process. Includes error handling if the file is not found. |
| **74-80** | `def open_neurology_module(self): script = os.path.join(..., "neurology.py")... subprocess.Popen(...)` | **`neurology.py` Launcher:** Defines the method to launch the `neurology.py` script as a new process. Includes error handling. |

---

## 3. `main.py` - Core PyTorch Model and Prediction Logic (Backend)

This script contains the definition of the cross-modal deep learning model and a robust function for batch prediction. The model is a hybrid network designed to handle both sequence (temporal) and kinematic (static) features.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| **8-11** | `class AtaxiaFusionNet(nn.Module): def __init__(self...): super().__init__()` | Defines the **AtaxiaFusionNet** class, inheriting from PyTorch's base module, and initializes hidden layer sizes. |
| **12-14** | `self.bilstm = nn.LSTM(..., bidirectional=True)` | **Sequence Branch:** Creates a **Bidirectional LSTM** layer to extract features from the time-series (sequence) input. |
| **15-19** | `self.mlp = nn.Sequential(...)` | **Kinematic Branch:** Creates a simple **Multi-Layer Perceptron (MLP)** with ReLU and Dropout to process the static feature vector. |
| **20-24** | `self.fusion = nn.Sequential(nn.Linear(..., fusion_hidden), ...)` | **Fusion Block:** Creates the final classification layers. Its input dimension is the sum of the BiLSTM output ($2 \times 64$) and the MLP output ($64$), demonstrating **feature concatenation**. |
| **28-29** | `_, (hn, _) = self.bilstm(x_seq); lstm_out = torch.cat((hn[-2], hn[-1]), dim=1)` | **Forward Pass:** Processes the sequence data and concatenates the forward and backward final hidden states. |
| **30-31** | `mlp_out = self.mlp(x_kin); combined = torch.cat((lstm_out, mlp_out), dim=1)` | Processes kinematic data and **concatenates** the sequence output and the kinematic output before passing to the fusion layer. |
| **36-49** | `def predict(test_path, ...): df = pd.read_csv(...) X = df.drop(columns=["label"])... X_scaled = scaler.fit_transform(X.values)` | **Prediction Function:** Loads the test data, removes the optional 'label' column, and **standardizes** the feature values using `StandardScaler`. |
| **51-57** | `seq_len = 5; pad_width = ...; X_scaled = np.pad(...) X_seq = X_scaled.reshape(...)` | **Preprocessing:** Pads the feature array to ensure its width is divisible by `seq_len=5`, then **reshapes** the 2D data into a 3D tensor suitable for the LSTM (Batch $\times$ Sequence Length $\times$ Features). |
| **64-67** | `model = AtaxiaFusionNet(input_dim_seq=...); model.load_state_dict(torch.load(model_path, ...))` | Initializes the model and loads the **pre-trained weights** from `ataxia_fusion_model.pth`. |
| **73-74** | `outputs = model(...); probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()` | Performs inference and converts the raw output logits into **probability scores** (confidence for the 'ataxia' class). |
| **77-87** | `df_original["Predicted_Label"] = [...]; df_original["Confidence"] = probs; df_original.to_csv(...)` | Appends predictions and confidence scores to the results, prints a summary, and **saves the final output** to `outputs/final_predictions.csv`. |

---

## 4. `neurology.py` & `dd.py` - Prediction Dashboards (GUIs)

These two dashboard scripts are functionally almost identical, both serving as a graphical frontend for the `AtaxiaFusionNet` prediction logic. `neurology.py` focuses on detailed, record-by-record output, while `dd.py` shows a top-10 preview and a summary.

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| **14-46** | `class AtaxiaFusionNet(nn.Module): ... ataxia_info = { ... }` | **Model/Info:** Defines the `AtaxiaFusionNet` (as in `main.py`) and a dictionary (`ataxia_info`) containing pre-canned **medical recommendation** text. |
| **51-70** | `class AtaxiaApp: def __init__(self, root): ... self.model_path = "outputs/ataxia_fusion_model.pth"` | **GUI Setup:** Initializes the dashboard, sets the model path, and creates the main UI elements: a **scrolled text area** for results and buttons for **Browse**, **Predict**, and **Show Graphs**. |
| **86-127** | `def predict_data(self): ...` | **Prediction Core:** This method contains the **entire prediction pipeline** (data loading, scaling, reshaping, model loading, and inference) cloned directly from `main.py`. |
| **137-142** | `(neurology.py only) for i, (pred, conf) in enumerate(zip(preds, probs)): if pred == 1: self.text_area.insert(tk.END, f"Record {i+1}: ATA XIA DETECTED ...")` | **Detailed Output (neurology.py):** Iterates over *all* records and prints a separate block of text for each subject, including the **full recommendation** if Ataxia is detected. |
| **129-140** | `(dd.py only) self.text_area.insert(tk.END, df_original.head(10).to_string(index=False)); if sum(preds) > 0: self.text_area.insert(tk.END, "--- Medical Recommendation ---")` | **Summary Output (dd.py):** Prints only the **first 10 rows** of predictions and then appends the **medical recommendation** if *any* case of Ataxia was detected in the batch. |
| **151-163** | `def show_graphs(self): plots = [...] for img_file in plots: img = plt.imread(img_file); plt.show()` | **Visualization:** Defines a function to load and display a set of **pre-generated PNG images** (e.g., `plot1_class_distribution.png`), simulating the visualization of results or dataset analysis. |

---

## 5. `Final Dataset Modeling.ipynb` - PyTorch Image Classification

This notebook output documents a comparative study of CNN models for multi-class **Retinal Disease Classification** on an image dataset. The key finding is the superior performance of transfer learning models (ResNet50, VGG19) over a custom CNN.

| Code Snippet / Output | Model | Key Step / Result |
| :--- | :--- | :--- |
| **`import torch, torchvision, ...`** | N/A | **Setup:** Imports PyTorch, `torchvision` (for models/datasets/transforms), `sklearn` (for metrics), and `matplotlib` (for plotting). |
| **`device = torch.device("cuda"...); data_dir = "Dataset"; num_classes = 8`** | N/A | **Configuration:** Sets the device, data path, and defines the problem as **8-class classification** (Age, Cataract, Diabetes, Glaucoma, Hypertension, Myopia, Normal, Other). |
| **`train_transform = transforms.Compose([...])`** | N/A | **Data Augmentation:** Defines a strong augmentation pipeline for training, including RandomResizeCrop, RandomRotation, ColorJitter, and GaussianBlur to prevent overfitting. |
| **`class CustomCNN(nn.Module): ...`** | Custom CNN | **Architecture:** Defines a simple, sequential **Custom CNN** with three convolutional blocks, pooling, and two fully connected layers. |
| **Training Output** | Custom CNN | **Performance:** Achieves a **Final Accuracy of 29.95%** on the evaluation set. The classification report shows very poor performance (e.g., $0.13$ F1-score for Diabetes), indicating the model failed to learn complex features from scratch. |
| **`resnet = models.resnet50(weights=...)`** | ResNet50 | **Transfer Learning:** Loads the **ResNet50** architecture, pre-trained on ImageNet, with initial layers frozen (`requires_grad = False`). |
| **`resnet.fc = nn.Sequential(...)`** | ResNet50 | **Fine-Tuning:** Replaces the final fully connected layer (`fc`) with a custom sequential block containing linear layers, BatchNorm1d, ReLU, and Dropout. This adapts the pre-trained features to the 8-class retinal task. |
| **Training Output** | ResNet50 | **Performance:** Achieves a **Final Accuracy of 93.62%**. The F1-scores are high across classes (e.g., $0.97$ for Normal, $0.92$ for Diabetes), demonstrating the effectiveness of transfer learning. |
| **`vgg19 = models.vgg19(weights=...)`** | VGG19 | **Transfer Learning:** Loads the **VGG19** architecture, replacing its classifier head for fine-tuning, a common technique for medical image analysis. |
| **Training Output** | VGG19 | **Performance:** Achieves the best result with a **Final Accuracy of 96.58%**. The classification report shows excellent performance, with most F1-scores above $0.95$. |
| **Comparison Plots** | N/A | **Visualization:** The generated plots compare the accuracy and loss curves over 25 epochs, clearly showing VGG19 as the top-performing model in this comparative benchmark. |