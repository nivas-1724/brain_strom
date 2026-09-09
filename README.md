# 🧠 NeuroScan AI — Brain Tumor Detection

A full-stack deep learning application for detecting brain tumors from MRI scans using **EfficientNetB0** transfer learning and **Grad-CAM** visualization.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange) ![Flask](https://img.shields.io/badge/Flask-2.3+-green) ![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **EfficientNetB0** | Transfer learning model trained on 7,000+ MRI images |
| 🔥 **Grad-CAM** | Heatmap highlighting the exact tumor region |
| 📊 **4-Class Detection** | Glioma, Meningioma, Pituitary, No Tumor |
| 📍 **Bounding Box** | Auto-computed from Grad-CAM activation map |
| 📄 **PDF Reports** | Professional diagnostic reports with patient info |
| 💻 **Premium Web UI** | Dark-mode glassmorphism design with drag & drop |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python setup_dataset.py
```
> Requires a Kaggle API key. See instructions inside the script.

### 3. Train the Model
```bash
python model/train.py
```
> Training takes ~30-60 min on GPU, ~3-4 hours on CPU.
> Model is saved to `model/saved/brain_tumor_model.h5`

### 4. Start the Server
```bash
python backend/app.py
```
> Open http://localhost:5000 in your browser.

---

## 📁 Project Structure

```
brain-tumor-detection/
├── model/
│   ├── train.py              # EfficientNetB0 training pipeline
│   ├── gradcam.py            # Grad-CAM heatmap generation
│   ├── predict.py            # Inference module
│   └── saved/
│       ├── brain_tumor_model.h5   # Trained model (after training)
│       └── class_indices.json
├── backend/
│   ├── app.py                # Flask REST API
│   └── report.py             # PDF report generator
├── frontend/
│   ├── index.html            # Web UI
│   ├── style.css             # Premium dark theme
│   └── app.js                # Frontend logic
├── dataset/                  # Downloaded dataset (after setup)
├── setup_dataset.py          # Dataset downloader
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
    ▼
EfficientNetB0 (ImageNet pretrained, frozen)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(512, relu) → Dropout(0.4)
    │
    ▼
Dense(256, relu) → Dropout(0.3)
    │
    ▼
Dense(4, softmax)  [Glioma | Meningioma | No Tumor | Pituitary]
```

**Training Strategy:**
1. Phase 1: Freeze base, train top layers (30 epochs)
2. Phase 2: Unfreeze top 30 layers of EfficientNetB0, fine-tune (10 epochs)

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Server health check |
| `POST` | `/api/predict` | Upload MRI → get prediction |
| `POST` | `/api/report`  | Generate PDF report |
| `GET`  | `/api/classes` | Get tumor class info |

### Example: POST /api/predict
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@brain_mri.jpg"
```

### Response:
```json
{
  "success": true,
  "prediction": "glioma",
  "display_name": "Glioma",
  "confidence": 94.3,
  "severity": "High",
  "color": "#ef4444",
  "scores": {
    "glioma":     {"confidence": 94.3},
    "meningioma": {"confidence": 2.1},
    "notumor":    {"confidence": 1.9},
    "pituitary":  {"confidence": 1.7}
  },
  "original_b64": "data:image/png;base64,...",
  "overlay_b64":  "data:image/png;base64,...",
  "heatmap_b64":  "data:image/png;base64,...",
  "bounding_box": {"x": 42.1, "y": 38.5, "width": 20.3, "height": 18.2},
  "characteristics": [...],
  "treatment": "..."
}
```

---

## 📊 Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle
- **Total Images**: ~7,000
- **Split**: Training (85%) / Validation (15%) / Testing (separate)

| Class | Training | Testing |
|-------|----------|---------|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |

---

## ⚠️ Medical Disclaimer

> This application is intended for **educational and research purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 📜 License

MIT License — Free to use for educational and research purposes.
