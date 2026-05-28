# ECG Emotion Recognition — Web-Based System

**Web-Based ECG Emotion Recognition with Automated Reporting Using Deep Learning and HRV Features**

> COMP6016 Final Project | Curtin University | May 2026  
> Student: Reza Rifat Akhlaque (22439437) | Supervisor: Muhammad Hasan

---

## Live Demo

**[https://rezarifat-ecg-emotion-detector.hf.space](https://rezarifat-ecg-emotion-detector.hf.space)**

No installation required — upload an ECG CSV file and get a result instantly.

---

## What This Project Does

This system classifies human emotions from raw ECG (electrocardiogram) signals using a hybrid deep learning model. It combines:

- A **1D CNN** for learning waveform shape features
- A **Bidirectional LSTM** for capturing temporal patterns
- An **HRV MLP** for processing 10 heart rate variability features

The model classifies ECG signals into **4 emotion classes**:
- Happy / Excited
- Angry / Stressed
- Calm / Relaxed
- Sad / Bored

**Results on DREAMER dataset:**
- Validation Accuracy: **88.8%**
- Macro F1-Score: **0.89**
- Model size: 266,276 parameters (~1.5 MB)
- Inference time: < 5 seconds on CPU

---

## Repository Structure

```
ecg-emotion-recognition/
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── predict.py         # ECG preprocessing and inference pipeline
│   ├── model.py           # HybridECGModel architecture
│   └── report_gen.py      # PDF report generation
├── frontend/
│   ├── index.html         # Web upload interface
│   └── styles.css
├── model/
│   ├── hybrid_ecg_model.pth   # Trained model weights
│   └── scaler.pkl             # Fitted HRV StandardScaler
├── training/
│   ├── train.py           # Training script (DREAMER dataset)
│   └── hrv_features.py    # HRV feature extraction utilities
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start — Option 1: Run Locally (Python)

### Requirements
- Python 3.10 or 3.11 recommended (3.12 requires torch>=2.2.0)
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/rezarifat/ecg-emotion-recognition.git
cd ecg-emotion-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 4. Open in browser
# http://localhost:8080
```

---

## Quick Start — Option 2: Run with Docker

```bash
# 1. Clone the repository
git clone https://github.com/rezarifat/ecg-emotion-recognition.git
cd ecg-emotion-recognition

# 2. Build the Docker image
docker build -t ecg-emotion-app .

# 3. Run the container
docker run -p 8080:8080 ecg-emotion-app

# 4. Open in browser
# http://localhost:8080
```

---

## How to Use

1. Open the web interface (localhost or HF Spaces URL)
2. Click **Upload ECG File** and select a `.csv` or `.dcm` file
3. The system will:
   - Segment the signal into 10-second windows
   - Apply bandpass filtering and normalisation
   - Extract 10 HRV features per window
   - Run the HybridECGModel
   - Return the predicted emotion and confidence scores
4. Download the auto-generated **PDF report**

### Input Format

| Format | Description |
|--------|-------------|
| CSV | Single column of ECG amplitude values, sampled at 256 Hz |
| DICOM (.dcm) | Standard clinical ECG format — waveform extracted automatically |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web frontend |
| POST | `/predict` | Upload ECG file, returns JSON result |
| GET | `/report/{uuid}` | Download PDF report |
| GET | `/health` | Health check |

### Example Response (`POST /predict`)

```json
{
  "prediction": "Happy/Excited",
  "confidence": {
    "Happy/Excited": 0.82,
    "Angry/Stressed": 0.07,
    "Calm/Relaxed": 0.06,
    "Sad/Bored": 0.05
  },
  "hrv_features": {
    "sdnn": 42.3,
    "rmssd": 31.1,
    "sd1": 22.0,
    "sd2": 58.7,
    "sd1_sd2_ratio": 0.375
  },
  "report_url": "/report/3f7a9c12-..."
}
```

---

## Model Architecture

```
ECG Signal (2560 samples)          HRV Features (10)
        │                                  │
   ┌────▼────┐                      ┌──────▼──────┐
   │  1D CNN  │    ┌──────────────┐  │  HRV MLP    │
   │  branch  │    │  BiLSTM      │  │  branch     │
   └────┬────┘    │  branch      │  └──────┬──────┘
        │          └──────┬───────┘         │
        └──────────────────┴────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Concat +   │
                    │  FC layers  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Softmax    │
                    │  (4 classes)│
                    └─────────────┘
```

**Key design decision:** Adding the SD1/SD2 HRV ratio feature raised Sad/Bored recall from 73% to 88%.

---

## Training

The model was trained on the **DREAMER** dataset (23 subjects, 9,640 windows, 4 classes).

To retrain:

```bash
# Requires DREAMER dataset (.mat file) — request from original authors
# Place dataset at training/DREAMER.mat

cd training
python train.py
# Trains for up to 72 epochs with early stopping (patience=10)
# Best weights saved to model/hybrid_ecg_model.pth
# Scaler saved to model/scaler.pkl
```

Training was done on Google Colab Pro (T4 GPU, ~35 minutes).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep learning | PyTorch |
| Web backend | FastAPI + Uvicorn |
| HRV extraction | NeuroKit2 |
| PDF reports | ReportLab |
| Frontend | HTML / CSS / JavaScript |
| Containerisation | Docker |
| Hosting | Hugging Face Spaces |

---

## Privacy and Data Handling

- Each upload is assigned a **UUID** — no personally identifiable information is stored
- All uploaded files and generated reports are **deleted after the HTTP response** is sent
- No ECG data is persisted to any database or disk storage
- Communication is over **HTTPS** on Hugging Face Spaces

---

## Known Limitations

- Tested on DREAMER (23 subjects, controlled lab setting) — real-world accuracy may vary
- Currently supports Lead II ECG only
- Signals shorter than 10 seconds cannot be classified
- Hugging Face Spaces may take 30–60 seconds to wake up after inactivity

---

## License

This project was developed as part of COMP6016 at Curtin University. The DREAMER dataset is not included and must be requested from the original authors separately.

---

*Reza Rifat Akhlaque | 22439437 | Curtin University | May 2026*
