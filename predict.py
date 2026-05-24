import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks, resample
from model import HybridECGModel
import joblib
import os

FS_MODEL = 256
WINDOW   = FS_MODEL * 10  # 2560 samples = 10 seconds

EMOTION_NAMES = ["Happy/Excited", "Angry/Stressed", "Calm/Relaxed", "Sad/Bored"]

EMOTION_DESC = {
    "Happy/Excited":  "High energy, positive emotional state detected.",
    "Angry/Stressed": "High arousal with negative valence detected.",
    "Calm/Relaxed":   "Low arousal, positive and relaxed state detected.",
    "Sad/Bored":      "Low energy, negative emotional state detected.",
}

# ── Load model once at startup ───────────────────────────────────
model = HybridECGModel(hrv_dim=10, num_classes=4)
model.load_state_dict(torch.load("ecg_emotion_model.pt", map_location="cpu"))
model.eval()

# ── Load HRV scaler ──────────────────────────────────────────────
if os.path.exists("hrv_scaler.pkl"):
    scaler = joblib.load("hrv_scaler.pkl")
else:
    scaler = None
    print("WARNING: hrv_scaler.pkl not found")


# ── Signal Loading ───────────────────────────────────────────────

def load_dicom(filepath):
    import pydicom
    ds = pydicom.dcmread(filepath)
    try:
        fs = float(ds.SamplingFrequency)
    except AttributeError:
        try:
            fs = float(ds[0x003a, 0x001a].value)
        except Exception:
            fs = 500.0
    try:
        sequence     = ds.WaveformSequence[0]
        channel_data = sequence.WaveformData
        n_channels   = sequence.NumberOfWaveformChannels
        n_samples    = sequence.NumberOfWaveformSamples
        raw           = np.frombuffer(channel_data, dtype=np.int16)
        signal_matrix = raw.reshape((n_samples, n_channels))
        ecg           = signal_matrix[:, 0].astype(np.float32)
        try:
            sensitivity = float(sequence.ChannelDefinitionSequence[0].ChannelSensitivity)
            ecg = ecg * sensitivity
        except Exception:
            pass
    except Exception as e:
        raise ValueError(f"Could not extract ECG from DICOM file: {e}")
    return ecg, fs


def load_csv(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    try:
        ecg = df.iloc[:, 0].values.astype(np.float32)
    except ValueError:
        df  = pd.read_csv(filepath, header=None, skiprows=1)
        ecg = df.iloc[:, 0].values.astype(np.float32)
    return ecg, FS_MODEL


def resample_to_model_fs(ecg, original_fs):
    if abs(original_fs - FS_MODEL) < 1:
        return ecg
    target_samples = int(len(ecg) * FS_MODEL / original_fs)
    return resample(ecg, target_samples).astype(np.float32)


# ── Preprocessing ────────────────────────────────────────────────

def bandpass_filter(signal):
    nyq = FS_MODEL / 2
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_hrv(ecg):
    """10 HRV features — matches training notebook exactly"""
    peaks, _ = find_peaks(ecg, distance=int(0.5 * FS_MODEL), height=np.mean(ecg))
    rr = np.diff(peaks) / FS_MODEL * 1000
    if len(rr) < 4:
        return None, {}

    mean_rr  = float(np.mean(rr))
    sdnn     = float(np.std(rr))
    rmssd    = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    hr       = float(60000 / mean_rr)
    pnn50    = float(np.mean(np.abs(np.diff(rr)) > 50) * 100)
    rr_range = float(np.max(rr) - np.min(rr))
    cv       = sdnn / mean_rr * 100
    sd1      = rmssd / np.sqrt(2)
    sd2      = float(np.sqrt(max(0, 2 * sdnn**2 - 0.5 * rmssd**2)))
    sd_ratio = sd1 / (sd2 + 1e-8)

    features_raw = np.array(
        [mean_rr, sdnn, rmssd, hr, pnn50, rr_range, cv, sd1, sd2, sd_ratio],
        dtype=np.float32
    )
    labels = {
        "Mean RR (ms)":     round(mean_rr, 1),
        "SDNN (ms)":        round(sdnn, 1),
        "RMSSD (ms)":       round(rmssd, 1),
        "Heart Rate (bpm)": round(hr, 1),
        "pNN50 (%)":        round(pnn50, 1),
        "RR Range (ms)":    round(rr_range, 1),
        "CV (%)":           round(cv, 1),
        "SD1 (ms)":         round(sd1, 1),
        "SD2 (ms)":         round(sd2, 1),
        "SD1/SD2":          round(sd_ratio, 3),
    }
    return features_raw, labels


# ── Main Prediction ──────────────────────────────────────────────

def predict_from_file(filepath: str):
    filepath = str(filepath)

    if filepath.lower().endswith('.dcm'):
        ecg_raw, fs = load_dicom(filepath)
    elif filepath.lower().endswith('.csv'):
        ecg_raw, fs = load_csv(filepath)
    else:
        raise ValueError("Unsupported file type. Please upload a .dcm or .csv file.")

    ecg_resampled = resample_to_model_fs(ecg_raw, fs)

    if len(ecg_resampled) < WINDOW:
        raise ValueError(
            f"ECG signal too short. Need at least 10 seconds "
            f"(got {len(ecg_resampled) / FS_MODEL:.1f}s)."
        )

    ecg_clean = bandpass_filter(ecg_resampled)
    mid    = len(ecg_clean) // 2
    start  = max(0, mid - WINDOW // 2)
    window = ecg_clean[start:start + WINDOW]

    hrv_raw, hrv_labels = extract_hrv(window)
    if hrv_raw is None:
        hrv_raw = np.zeros(10, dtype=np.float32)

    # Apply StandardScaler — must match training
    if scaler is not None:
        hrv_scaled = scaler.transform(hrv_raw.reshape(1, -1))[0].astype(np.float32)
    else:
        hrv_scaled = hrv_raw

    window_norm = (window - window.mean()) / (window.std() + 1e-8)

    ecg_t = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hrv_t = torch.tensor(hrv_scaled,  dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(ecg_t, hrv_t)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
        pred   = int(probs.argmax())

    return {
        "emotion":           EMOTION_NAMES[pred],
        "description":       EMOTION_DESC[EMOTION_NAMES[pred]],
        "confidence":        round(float(probs[pred]) * 100, 1),
        "all_probs":         {EMOTION_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(4)},
        "hrv_features":      hrv_labels,
        "signal_preview":    window_norm[:512].tolist(),
        "sampling_rate":     fs,
        "signal_length_sec": round(len(ecg_resampled) / FS_MODEL, 1),
    }
