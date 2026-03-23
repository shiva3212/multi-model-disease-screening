import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8
image_size = 224
model_path = "vgg19_retina.pth"

class_names = ['Age', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Other']

# ============================================================
# DISEASE INFORMATION
# ============================================================
disease_info = {
    "Age": {
        "description": "Age-related changes in the retina often involve degeneration of photoreceptor cells, thickening of Bruch’s membrane, and reduced macular pigment density. These changes can lead to age-related macular degeneration (AMD), dry eyes, decreased contrast sensitivity, and slower adaptation to low light. Over time, structural and metabolic alterations in retinal tissue impair central vision and may cause permanent loss of acuity if untreated.",
        "risk": "Older adults (particularly over 60 years), oxidative stress due to free radicals, diets lacking in antioxidants, long-term UV light exposure, smoking, hypertension, and excessive alcohol consumption.",
        "prevention": "Maintain a diet high in antioxidants (vitamins A, C, E, lutein, and zeaxanthin), consume omega-3 fatty acids from fish, wear UV-protective sunglasses, avoid tobacco smoke, and engage in regular physical activity to enhance blood circulation to ocular tissues.",
        "treatment": "Routine ophthalmic checkups for early detection of AMD or other degenerative changes, supplementation with macular pigments (lutein, zeaxanthin), use of lubricating eye drops for dryness, and prescription lenses to correct age-related refractive changes."
    },
    "Cataract": {
        "description": "A cataract is characterized by opacification of the eye’s natural crystalline lens, resulting in scattered light transmission and blurred or faded vision. It typically develops slowly as proteins within the lens denature and clump together. Advanced cases can significantly impair night vision, color differentiation, and glare tolerance.",
        "risk": "Aging, diabetes mellitus, excessive UV exposure, prolonged corticosteroid use, smoking, eye trauma, and a positive family history of cataract formation.",
        "prevention": "Wear UV-blocking sunglasses, manage chronic conditions such as diabetes, limit exposure to cigarette smoke and alcohol, and consume foods rich in antioxidants and carotenoids to maintain lens clarity.",
        "treatment": "Definitive management involves surgical removal of the clouded lens (phacoemulsification) followed by implantation of an intraocular lens (IOL). Early-stage cases may benefit temporarily from stronger spectacles, magnifying lenses, and improved ambient lighting."
    },
    "Diabetes": {
        "description": "Diabetic retinopathy arises when persistently elevated blood glucose damages retinal capillaries, leading to microaneurysms, leakage, and ischemia. In advanced stages, new fragile blood vessels proliferate (proliferative retinopathy), potentially causing vitreous hemorrhage or retinal detachment. Macular edema frequently accompanies this condition, resulting in central vision loss.",
        "risk": "Prolonged uncontrolled diabetes, coexisting hypertension or hyperlipidemia, obesity, poor glycemic control (HbA1c >7%), and disease duration over 10 years.",
        "prevention": "Strict control of blood sugar and blood pressure, adherence to an anti-inflammatory diet rich in fiber and omega-3 fats, regular dilated fundus examinations (at least annually), and maintaining an appropriate body weight.",
        "treatment": "Laser photocoagulation to seal leaking vessels, intravitreal anti-VEGF injections to reduce neovascularization, intravitreal steroids to manage macular edema, and systemic management through optimized diabetes and BP medications."
    },
    "Glaucoma": {
        "description": "Glaucoma includes a group of diseases characterized by progressive optic neuropathy caused primarily by elevated intraocular pressure (IOP) or impaired optic nerve blood flow. The most common type, open-angle glaucoma, progresses asymptomatically until significant vision loss occurs. Damage to retinal ganglion cells and optic nerve fibers leads to peripheral vision loss (tunnel vision) and, at later stages, complete blindness.",
        "risk": "Family history of glaucoma, age over 40 years, elevated IOP, diabetes, severe myopia, long-term corticosteroid use, and decreased corneal thickness.",
        "prevention": "Regular IOP monitoring (especially for high-risk individuals), maintaining healthy blood flow through moderate exercise, avoiding excessive caffeine, and limiting steroid-based eye drops unless medically necessary.",
        "treatment": "Topical medications such as prostaglandin analogs, beta blockers, or carbonic anhydrase inhibitors to lower IOP; laser trabeculoplasty; and filtration surgery (trabeculectomy or tube shunt implantation) for advanced or resistant cases."
    },
    "Hypertension": {
        "description": "Hypertensive retinopathy develops when prolonged high arterial pressure induces vascular constriction, thickening, and eventual leakage within retinal vessels. Chronic exposure damages the arterioles, leading to flame-shaped hemorrhages, cotton-wool spots, retinal edema, and in severe cases, optic disc swelling (papilledema).",
        "risk": "Long-standing hypertension, obesity, sedentary lifestyle, excessive salt intake, smoking, stress, and concomitant disorders such as atherosclerosis or diabetes.",
        "prevention": "Maintain optimal blood pressure levels through medication and lifestyle changes, reduce salt intake, manage stress with breathing or meditation practices, and consume potassium-rich fruits like bananas and avocados.",
        "treatment": "Systematic control of systemic blood pressure using antihypertensive classes (ACE inhibitors, calcium channel blockers, beta blockers). Regular ophthalmic checkups ensure early detection and prevention of irreversible retinal damage."
    },
    "Myopia": {
        "description": "Myopia (nearsightedness) occurs when the eyeball elongates or the cornea curves excessively, causing light rays to focus before the retina. It leads to blurry distance vision while close objects remain clear. Progressive or pathological myopia may induce retinal thinning or detachment due to structural stress.",
        "risk": "Genetic inheritance, prolonged screen exposure or near work without breaks, lack of outdoor sunlight exposure (reduced dopamine signaling in retina), and early onset during childhood.",
        "prevention": "Adopt the 20-20-20 rule (every 20 minutes, look 20 feet away for 20 seconds), spend at least 2 hours outdoors daily, ensure proper reading distance and lighting, and limit recreational screen use.",
        "treatment": "Corrective eyeglasses or contact lenses, orthokeratology lenses for mild cases, and refractive surgeries such as LASIK, SMILE, or PRK for permanent correction. Regular eye exams are crucial to monitor axial elongation in children."
    },
    "Normal": {
        "description": "A normal retina appears healthy with a well-defined optic disc, bright macula, clear blood vessels, and uniform pigmentation. There are no signs of hemorrhages, exudates, or vessel constriction. Good retinal health reflects balanced ocular nutrition, proper hydration, and effective blood flow regulation.",
        "risk": "No significant risk factors if general health is maintained, though gradual changes can occur with age or exposure to eye strain and UV light.",
        "prevention": "Maintain a balanced diet rich in eye-supportive nutrients, avoid excessive screen fatigue, practice proper eye hygiene, and schedule annual eye examinations even in the absence of symptoms.",
        "treatment": "No medical treatment necessary. Continue preventive care such as eye relaxation exercises, hydration, and maintaining optimal light conditions during reading or work."
    },
    "Other": {
        "description": "This category includes miscellaneous or rare retinal abnormalities not confined to common pathologies. Conditions may range from inherited retinal dystrophies (e.g., retinitis pigmentosa) to infections (toxoplasmosis), inflammatory disorders (uveitis), traumatic injuries, or ischemic events. Each has distinct pathology, symptoms, and visual implications.",
        "risk": "Viral or bacterial infections, ocular trauma, hereditary genetic disorders, systemic autoimmune diseases, or exposure to toxins and radiation.",
        "prevention": "Adopt good eye safety practices, ensure protective eyewear during hazardous activities, promptly treat systemic infections, and undergo genetic counseling if family retinal disorders exist.",
        "treatment": "Management varies depending on cause — antibiotics or antivirals for infections, corticosteroids for inflammation, anti-VEGF or immunomodulatory drugs for vascular or autoimmune causes, and surgical interventions for retinal tears or detachment."
    }
}

# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    for param in vgg19.parameters():
        param.requires_grad = False

    in_features = vgg19.classifier[0].in_features
    vgg19.classifier = nn.Sequential(
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

    vgg19.load_state_dict(torch.load(model_path, map_location=device))
    vgg19.to(device)
    vgg19.eval()
    print(" Model loaded successfully!")
    return vgg19


# ============================================================
# IMAGE TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100
    return predicted_class, confidence


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    model = load_model()
    input_path = input("Enter image path or folder: ").strip()
    results = []

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(input_path, file)
                pred_class, conf = predict_image(model, full_path)
                info = disease_info.get(pred_class, {})
                print(f"\n {file}")
                print(f" Predicted Class: {pred_class} ({conf:.2f}%)")
                print(f" Description: {info.get('description', 'N/A')}")
                print(f" Risk Factors: {info.get('risk', 'N/A')}")
                print(f" Prevention: {info.get('prevention', 'N/A')}")
                print(f" Treatment: {info.get('treatment', 'N/A')}")
                results.append({"filename": file, "predicted_class": pred_class, "confidence": conf})

    elif os.path.isfile(input_path):
        pred_class, conf = predict_image(model, input_path)
        info = disease_info.get(pred_class, {})
        print(f"\n Image: {os.path.basename(input_path)}")
        print(f" Predicted Class: {pred_class} ({conf:.2f}%)")
        print(f" Description: {info.get('description', 'N/A')}")
        print(f" Risk Factors: {info.get('risk', 'N/A')}")
        print(f" Prevention: {info.get('prevention', 'N/A')}")
        print(f" Treatment: {info.get('treatment', 'N/A')}")
        results.append({"filename": os.path.basename(input_path), "predicted_class": pred_class, "confidence": conf})

    else:
        print(" Invalid path!")

    if results:
        pd.DataFrame(results).to_csv("vgg19_predictions.csv", index=False)
        print("\n Predictions saved to 'vgg19_predictions.csv'")
