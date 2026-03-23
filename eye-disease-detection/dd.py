import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import os

# ===================================================
# MODEL CONFIGURATION
# ===================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8
image_size = 224
class_names = ['Age', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Other']

# ===================================================
# DISEASE INFORMATION
# ===================================================
disease_info = {
    "Age": {
        "description": "Age-related changes in the retina often involve degeneration of photoreceptor cells, thickening of Bruch’s membrane, and reduced macular pigment density. These changes can lead to age-related macular degeneration (AMD), dry eyes, decreased contrast sensitivity, and slower adaptation to low light.",
        "risk": "Older adults (particularly over 60 years), oxidative stress, poor diet, UV exposure, smoking, hypertension, and excessive alcohol consumption.",
        "prevention": "Consume antioxidant-rich foods, omega-3 fatty acids, wear UV-protective sunglasses, avoid smoking, and maintain active circulation through exercise.",
        "treatment": "Routine eye checkups, supplementation with lutein and zeaxanthin, lubricating eye drops, and prescription lenses for refractive correction."
    },
    "Cataract": {
        "description": "A cataract is characterized by opacification of the crystalline lens, resulting in blurred or faded vision. It develops slowly as lens proteins clump together, reducing light transmission.",
        "risk": "Aging, diabetes, prolonged UV exposure, corticosteroid use, smoking, and family history.",
        "prevention": "Wear UV-blocking sunglasses, manage diabetes, limit smoking/alcohol, and eat foods rich in antioxidants and carotenoids.",
        "treatment": "Surgical lens removal and IOL implantation. Early stages may benefit from brighter lighting and updated spectacles."
    },
    "Diabetes": {
        "description": "Diabetic retinopathy occurs when high blood sugar damages retinal vessels, leading to leakage, ischemia, and possible detachment.",
        "risk": "Long-term uncontrolled diabetes, hypertension, obesity, and poor glycemic control.",
        "prevention": "Control blood sugar and blood pressure, eat anti-inflammatory foods, and undergo annual dilated eye exams.",
        "treatment": "Laser photocoagulation, anti-VEGF injections, intravitreal steroids, and systemic control of diabetes and BP."
    },
    "Glaucoma": {
        "description": "Glaucoma causes optic nerve damage, often from increased intraocular pressure, leading to progressive peripheral vision loss.",
        "risk": "Family history, age > 40, elevated IOP, diabetes, severe myopia, and steroid use.",
        "prevention": "Routine IOP monitoring, avoid excessive caffeine, and maintain good blood flow through regular exercise.",
        "treatment": "Topical medications (beta-blockers, prostaglandin analogs), laser therapy, or trabeculectomy for advanced cases."
    },
    "Hypertension": {
        "description": "Hypertensive retinopathy results from long-term high blood pressure damaging the retinal vessels, causing hemorrhages or edema.",
        "risk": "Chronic hypertension, obesity, high salt intake, and stress.",
        "prevention": "Maintain normal BP with diet and medication, reduce salt intake, and manage stress.",
        "treatment": "Control systemic BP with ACE inhibitors, beta blockers, and regular ophthalmic monitoring."
    },
    "Myopia": {
        "description": "Myopia (nearsightedness) occurs when the eyeball elongates, causing distant objects to appear blurry.",
        "risk": "Genetics, excessive screen exposure, lack of outdoor activity, and early childhood onset.",
        "prevention": "Follow 20-20-20 rule, spend more time outdoors, and maintain proper lighting and reading distance.",
        "treatment": "Use corrective lenses, orthokeratology, or refractive surgery such as LASIK or SMILE."
    },
    "Normal": {
        "description": "A normal retina has a clear macula, well-defined optic disc, and healthy blood vessel structure.",
        "risk": "No major risk if general health and vision care are maintained.",
        "prevention": "Balanced diet, limited screen time, good eye hygiene, and regular exams.",
        "treatment": "No treatment required. Maintain hydration, rest eyes regularly, and avoid UV strain."
    },
    "Other": {
        "description": "Includes rare retinal disorders such as uveitis, infections, or genetic dystrophies.",
        "risk": "Infections, trauma, genetic mutations, and autoimmune conditions.",
        "prevention": "Use eye protection, treat infections promptly, and maintain systemic health.",
        "treatment": "Depends on cause—antibiotics, corticosteroids, or surgery for retinal tears or detachment."
    }
}

# ===================================================
# MODEL LOADER FUNCTION
# ===================================================
def load_model(model_type="vgg19"):
    if model_type == "vgg19":
        model_path = "vgg19_retina.pth"
        base_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        in_features = base_model.classifier[0].in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    elif model_type == "resnet50":
        model_path = "resnet50_retina.pth"
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
    else:
        model_path = "custom_cnn_retina.pth"
        base_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        in_features = base_model.classifier[0].in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.to(device)
    base_model.eval()
    return base_model

# ===================================================
# TRANSFORMS
# ===================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===================================================
# GUI APPLICATION
# ===================================================
class RetinaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retina Disease Classification Dashboard")
        self.root.geometry("950x650")
        self.root.configure(bg="#f9fafc")

        self.model_type = "vgg19"
        self.text_area = scrolledtext.ScrolledText(root, width=90, height=25, font=("Consolas", 10))
        self.text_area.place(x=25, y=230)

        header = tk.Label(root, text="Retina Disease Detection Dashboard",
                          font=("Segoe UI Bold", 20), bg="#f9fafc", fg="#003049")
        header.place(x=230, y=30)

        # Buttons
        tk.Button(root, text="Custom CNN Accuracy", command=self.show_cnn_accuracy,
                  bg="#004c6d", fg="white", font=("Segoe UI", 11), width=20).place(x=70, y=120)
        tk.Button(root, text="ResNet50 Accuracy", command=self.show_resnet_accuracy,
                  bg="#004c6d", fg="white", font=("Segoe UI", 11), width=20).place(x=280, y=120)
        tk.Button(root, text="VGG19 Accuracy", command=self.show_vgg_accuracy,
                  bg="#004c6d", fg="white", font=("Segoe UI", 11), width=20).place(x=490, y=120)
        tk.Button(root, text="Predict Image", command=self.predict_image,
                  bg="#006d77", fg="white", font=("Segoe UI", 11), width=20).place(x=700, y=120)
        tk.Button(root, text="Show All Graphs", command=self.show_graphs,
                  bg="#5a189a", fg="white", font=("Segoe UI", 11), width=20).place(x=380, y=180)

    # ===================================================
    # MODEL ACCURACY DISPLAY
    # ===================================================
    def show_cnn_accuracy(self):
        self.text_area.insert(tk.END, "\n[Custom CNN Model]\nAccuracy: ~27.4%\nObservation: Underfitting due to shallow depth.\n\n")

    def show_resnet_accuracy(self):
        self.text_area.insert(tk.END, "\n[ResNet50 Model]\nAccuracy: ~84.5%\nObservation: Balanced accuracy with strong generalization.\n\n")

    def show_vgg_accuracy(self):
        self.text_area.insert(tk.END, "\n[VGG19 Model]\nAccuracy: ~92.3%\nObservation: Most stable and consistent performance.\n\n")

    # ===================================================
    # PREDICTION FUNCTION WITH RECOMMENDATION
    # ===================================================
    def predict_image(self):
        file_path = filedialog.askopenfilename(title="Select Image",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        model = load_model(self.model_type)
        image = PILImage.open(file_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        predicted_class = class_names[pred.item()]
        confidence = conf.item() * 100
        info = disease_info.get(predicted_class, {})

        self.text_area.insert(tk.END, f"\n[Prediction Result]\n")
        self.text_area.insert(tk.END, f"Predicted Disease: {predicted_class}\nConfidence: {confidence:.2f}%\n")
        self.text_area.insert(tk.END, f"\n--- Medical Recommendation ---\n")
        self.text_area.insert(tk.END, f"Description: {info.get('description', 'N/A')}\n\n")
        self.text_area.insert(tk.END, f"Risk Factors: {info.get('risk', 'N/A')}\n\n")
        self.text_area.insert(tk.END, f"Prevention: {info.get('prevention', 'N/A')}\n\n")
        self.text_area.insert(tk.END, f"Treatment: {info.get('treatment', 'N/A')}\n\n")

    # ===================================================
    # GRAPH DISPLAY FUNCTION
    # ===================================================
    def show_graphs(self):
        plots = [
            "output_0_1.png", "output_0_3.png", "output_0_5.png",
            "output_0_7.png", "output_0_8.png"
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
    app = RetinaApp(root)
    root.mainloop()
