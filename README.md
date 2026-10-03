## Technology Stack
- **Language**: Python 3.x
- **GUI Framework**: Tkinter
- **Deep Learning**: PyTorch
- **Models**: VGG19, ResNet50, Custom CNN
- **Image Processing**: OpenCV, PIL

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/multi-model-disease-screening.git
cd multi-model-disease-screening
```

2. Install dependencies:
```bash
cd neurological-screening
pip install -r requirements.txt
```

3. Download pre-trained models (see Models section below)

## Usage

### Eye Disease Detection
```bash
cd eye-disease-detection
python main.py
```

### Neurological Screening
```bash
cd neurological-screening
python main.py
```

## Models

Pre-trained model files are not included in this repository due to file size constraints (>100MB each).

**Available models:**
- Custom CNN (206 MB)
- VGG19 (132 MB)
- ResNet50 (99 MB)

**To use the system:**
1. Train models using the provided training scripts: `Train.py`
2. Or contact for pre-trained model weights

## Dataset

Datasets are not included due to privacy and size constraints. The system was trained on:
- Eye disease image datasets (categorized by disease type)
- Neurological disorder clinical data

For research purposes, datasets can be obtained from:
- [Kaggle Medical Datasets]
- [UCI Machine Learning Repository]

## Documentation

Detailed code explanations are available in:
- `neurological-screening/Section-Wise Code Explanation.pdf`
- Individual module README files

## Results

Model performance metrics and visualizations are available in the Jupyter notebooks:
- `eye-disease-detection/Final Dataset Modeling.ipynb`
- `neurological-screening/Final Dataset Modeling.ipynb`

## Disclaimer

⚠️ **Important**: This is a research/educational project demonstrating AI applications in healthcare. It is **NOT** intended for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Future Improvements

- Integration of both modules into unified interface
- Model optimization for real-time inference
- Deployment as web application
- Additional disease categories
- Enhanced explainability features (Grad-CAM, attention maps)

## Contributing

This is an academic project. Feedback and suggestions are welcome via issues.

## License

MIT License

## Contact

For questions or feedback, please open an issue on GitHub or contact shivatandure@gmail.com
---

**Note**: This project demonstrates the application of deep learning in healthcare diagnostics and serves as an educational resource for medical AI research.
