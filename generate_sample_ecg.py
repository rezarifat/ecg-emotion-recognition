"""
ECG Sample Data Generator
=========================
Run this script on your PC to generate sample test files.
It creates:
  1. sample_ecg_happy.csv
  2. sample_ecg_stressed.csv
  3. sample_ecg_calm.csv
  4. sample_ecg_sad.csv
  5. sample_ecg.dcm  (DICOM format)

Usage:
    python generate_sample_ecg.py

Requirements:
    pip install numpy scipy pydicom
"""

import numpy as np
from scipy.signal import resample
import struct, os

FS  = 256       # 256 Hz sampling rate (matches our model)
DUR = 30        # 30 seconds of ECG data
N   = FS * DUR  # total samples

# ─────────────────────────────────────────────
# ECG Simulation Helper
# ─────────────────────────────────────────────
def simulate_ecg(heart_rate=70, hrv_sdnn=50, noise=0.02, duration=30, fs=256):
    """
    Simulate a realistic ECG signal with PQRST waves.
    heart_rate : beats per minute
    hrv_sdnn   : HRV standard deviation in ms (higher = more variability)
    noise      : amplitude of random noise
    """
    n_samples  = duration * fs
    t          = np.linspace(0, duration, n_samples)
    ecg        = np.zeros(n_samples)

    # Base RR interval in samples
    rr_base = int(fs * 60 / heart_rate)

    # Generate beat positions with HRV variation
    beat_positions = []
    pos = rr_base
    while pos < n_samples - rr_base:
        jitter = int(np.random.normal(0, hrv_sdnn / 1000 * fs))
        pos   += rr_base + jitter
        if 0 < pos < n_samples:
            beat_positions.append(pos)

    # PQRST wave template
    def pqrst(t_local):
        """Generate one PQRST complex"""
        wave = np.zeros_like(t_local)
        # P wave (atrial depolarisation)
        p_mask = (t_local > -0.12) & (t_local < -0.02)
        wave[p_mask] += 0.15 * np.exp(-((t_local[p_mask] + 0.07)**2) / 0.001)
        # Q wave
        q_mask = (t_local > -0.02) & (t_local < 0.0)
        wave[q_mask] -= 0.10 * np.exp(-((t_local[q_mask] + 0.01)**2) / 0.00005)
        # R wave (main peak)
        r_mask = (t_local > -0.01) & (t_local < 0.01)
        wave[r_mask] += 1.00 * np.exp(-((t_local[r_mask])**2) / 0.00003)
        # S wave
        s_mask = (t_local > 0.01) & (t_local < 0.04)
        wave[s_mask] -= 0.15 * np.exp(-((t_local[s_mask] - 0.02)**2) / 0.0001)
        # T wave (ventricular repolarisation)
        t_mask = (t_local > 0.05) & (t_local < 0.25)
        wave[t_mask] += 0.25 * np.exp(-((t_local[t_mask] - 0.15)**2) / 0.003)
        return wave

    # Place PQRST complex at each beat position
    for bp in beat_positions:
        for i in range(n_samples):
            t_local = (i - bp) / fs
            if -0.15 < t_local < 0.35:
                ecg[i] += pqrst(np.array([t_local]))[0]

    # Add baseline wander + noise
    baseline = 0.05 * np.sin(2 * np.pi * 0.05 * t)
    ecg     += baseline + np.random.normal(0, noise, n_samples)

    return ecg.astype(np.float32)


# ─────────────────────────────────────────────
# Define 4 emotion profiles
# ─────────────────────────────────────────────
# Heart rate and HRV vary per emotional state:
#   Happy/Excited  → elevated HR, moderate HRV
#   Angry/Stressed → high HR, low HRV (sympathetic dominance)
#   Calm/Relaxed   → low HR, high HRV (parasympathetic dominance)
#   Sad/Bored      → low HR, low HRV

profiles = {
    "happy":   dict(heart_rate=85,  hrv_sdnn=55,  noise=0.025),
    "stressed":dict(heart_rate=100, hrv_sdnn=20,  noise=0.035),
    "calm":    dict(heart_rate=60,  hrv_sdnn=90,  noise=0.015),
    "sad":     dict(heart_rate=62,  hrv_sdnn=25,  noise=0.020),
}

print("Generating sample ECG files...\n")

# ─────────────────────────────────────────────
# 1. Save as CSV files
# ─────────────────────────────────────────────
for name, params in profiles.items():
    ecg      = simulate_ecg(**params)
    filename = f"sample_ecg_{name}.csv"
    np.savetxt(filename, ecg, delimiter=",", fmt="%.6f",
               header="ecg_amplitude", comments="")
    print(f"  ✅ {filename}  ({len(ecg)} samples, {len(ecg)/FS:.0f}s, HR~{params['heart_rate']}bpm)")

# ─────────────────────────────────────────────
# 2. Save as DICOM file (happy profile)
# ─────────────────────────────────────────────
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid
    import datetime

    ecg = simulate_ecg(**profiles["happy"])

    # Scale to int16
    scale       = 1000.0
    ecg_int16   = (ecg * scale).astype(np.int16)
    raw_bytes   = ecg_int16.tobytes()

    # Build DICOM dataset
    ds = FileDataset("sample_ecg.dcm", {}, is_implicit_VR=False, is_little_endian=True)

    # Required DICOM metadata
    ds.file_meta                         = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID       = pydicom.uid.ExplicitVRLittleEndian

    ds.SOPClassUID    = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.SOPInstanceUID = generate_uid()
    ds.Modality       = "ECG"
    ds.PatientName    = "Sample^Patient"
    ds.PatientID      = "ECG001"
    ds.StudyDate      = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime      = datetime.datetime.now().strftime("%H%M%S")

    # Waveform sequence
    waveform_item                        = Dataset()
    waveform_item.NumberOfWaveformChannels = 1
    waveform_item.NumberOfWaveformSamples  = len(ecg_int16)
    waveform_item.SamplingFrequency        = str(FS)
    waveform_item.WaveformData             = raw_bytes
    waveform_item.WaveformBitsAllocated    = 16
    waveform_item.WaveformSampleInterpretation = "SS"

    # Channel definition
    ch_def                   = Dataset()
    ch_def.ChannelSensitivity = "1.0"
    ch_def.ChannelLabel       = "I"
    waveform_item.ChannelDefinitionSequence = Sequence([ch_def])

    ds.WaveformSequence = Sequence([waveform_item])

    # Add sampling frequency also at top level for compatibility
    ds.add_new([0x003a, 0x001a], 'DS', str(FS))

    pydicom.dcmwrite("sample_ecg.dcm", ds)
    print(f"\n  ✅ sample_ecg.dcm  (DICOM format, Happy/Excited profile)")

except ImportError:
    print("\n  ⚠️  pydicom not installed — skipping DICOM file.")
    print("     Run: pip install pydicom")
    print("     Then re-run this script to generate the .dcm file.")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Done! Files created:

  CSV files (upload any of these):
    sample_ecg_happy.csv     → expect: Happy/Excited
    sample_ecg_stressed.csv  → expect: Angry/Stressed
    sample_ecg_calm.csv      → expect: Calm/Relaxed
    sample_ecg_sad.csv       → expect: Sad/Bored

  DICOM file:
    sample_ecg.dcm           → expect: Happy/Excited

Upload these at http://127.0.0.1:8000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")