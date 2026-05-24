import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks, resample
from model import HybridECGModel

FS_MODEL = 256
WINDOW   = FS_MODEL * 10  # 2560 samples = 10 seconds

EMOTION_NAMES = ["Happy/Excited", "Angry/Stressed", "Calm/Relaxed", "Sad/Bored"]

EMOTION_DESC = {
    "Happy/Excited":  "High energy, positive emotional state detected.",
    "Angry/Stressed": "High arousal with negative valence detected.",
    "Calm/Relaxed":   "Low arousal, positive and relaxed state detected.",
    "Sad/Bored":      "Low energy, negative emotional state detected.",
}

# Load model once at startup
model = HybridECGModel()
model.load_state_dict(torch.load("ecg_emotion_model.pt", map_location="cpu"))
model.eval()


# ── Signal Loading ───────────────────────────────────────────

def load_dicom(filepath):
    """Extract ECG waveform from a DICOM file"""
    import pydicom
    ds = pydicom.dcmread(filepath)

    # Get sampling frequency from DICOM metadata
    try:
        fs = float(ds.SamplingFrequency)
    except AttributeError:
        try:
            fs = float(ds[0x003a, 0x001a].value)
        except Exception:
            fs = 500.0  # common default for medical ECGs

    # Extract waveform data
    try:
        sequence   = ds.WaveformSequence[0]
        channel_data = sequence.WaveformData
        n_channels   = sequence.NumberOfWaveformChannels
        n_samples    = sequence.NumberOfWaveformSamples

        raw            = np.frombuffer(channel_data, dtype=np.int16)
        signal_matrix  = raw.reshape((n_samples, n_channels))
        ecg            = signal_matrix[:, 0].astype(np.float32)

        # Apply DICOM scaling if present
        try:
            sensitivity = float(sequence.ChannelDefinitionSequence[0].ChannelSensitivity)
            ecg = ecg * sensitivity
        except Exception:
            pass

    except Exception as e:
        raise ValueError(f"Could not extract ECG from DICOM file: {e}")

    return ecg, fs


def load_csv(filepath):
    """Load ECG from a single-column CSV"""
    import pandas as pd
    df = pd.read_csv(filepath)
    try:
        ecg = df.iloc[:, 0].values.astype(np.float32)
    except ValueError:
        df  = pd.read_csv(filepath, header=None, skiprows=1)
        ecg = df.iloc[:, 0].values.astype(np.float32)
    return ecg, FS_MODEL


def resample_to_model_fs(ecg, original_fs):
    """Resample ECG to 256 Hz if needed"""
    if abs(original_fs - FS_MODEL) < 1:
        return ecg
    target_samples = int(len(ecg) * FS_MODEL / original_fs)
    return resample(ecg, target_samples).astype(np.float32)


# ── Preprocessing ────────────────────────────────────────────

def bandpass_filter(signal):
    nyq = FS_MODEL / 2
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_hrv(ecg):
    peaks, _ = find_peaks(ecg, distance=int(0.5 * FS_MODEL), height=np.mean(ecg))
    rr = np.diff(peaks) / FS_MODEL * 1000  # ms
    if len(rr) < 3:
        return None, {}

    features = np.array([
        np.mean(rr),
        np.std(rr),
        np.sqrt(np.mean(np.diff(rr) ** 2)),
        60000 / np.mean(rr),
        np.mean(np.abs(np.diff(rr)) > 50) * 100,
        np.max(rr) - np.min(rr),
    ], dtype=np.float32)

    labels = {
        "Mean RR (ms)":     round(float(np.mean(rr)), 1),
        "SDNN (ms)":        round(float(np.std(rr)), 1),
        "RMSSD (ms)":       round(float(np.sqrt(np.mean(np.diff(rr) ** 2))), 1),
        "Heart Rate (bpm)": round(float(60000 / np.mean(rr)), 1),
        "pNN50 (%)":        round(float(np.mean(np.abs(np.diff(rr)) > 50) * 100), 1),
        "RR Range (ms)":    round(float(np.max(rr) - np.min(rr)), 1),
    }
    return features, labels


# ── Main Prediction ──────────────────────────────────────────

def predict_from_file(filepath: str):
    """Main entry point — auto-detects file type"""
    filepath = str(filepath)

    if filepath.lower().endswith('.dcm'):
        ecg_raw, fs = load_dicom(filepath)
    elif filepath.lower().endswith('.csv'):
        ecg_raw, fs = load_csv(filepath)
    else:
        raise ValueError("Unsupported file type. Please upload a .dcm or .csv file.")

    # Resample to 256 Hz if needed
    ecg_resampled = resample_to_model_fs(ecg_raw, fs)

    if len(ecg_resampled) < WINDOW:
        raise ValueError(
            f"ECG signal too short. Need at least 10 seconds "
            f"(got {len(ecg_resampled) / FS_MODEL:.1f}s)."
        )

    # Filter
    ecg_clean = bandpass_filter(ecg_resampled)

    # Use middle 10s window
    mid    = len(ecg_clean) // 2
    start  = max(0, mid - WINDOW // 2)
    window = ecg_clean[start:start + WINDOW]

    # HRV
    hrv_arr, hrv_labels = extract_hrv(window)
    if hrv_arr is None:
        hrv_arr = np.zeros(6, dtype=np.float32)

    # Normalize
    window_norm = (window - window.mean()) / (window.std() + 1e-8)

    # Model inference
    ecg_t = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hrv_t = torch.tensor(hrv_arr,     dtype=torch.float32).unsqueeze(0)

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
