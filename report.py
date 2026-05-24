import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import os

# Colors per emotion (R, G, B)
BAR_COLORS = {
    "Happy/Excited":  (241, 196,  15),
    "Angry/Stressed": (231,  76,  60),
    "Calm/Relaxed":   ( 46, 204, 113),
    "Sad/Bored":      ( 52, 152, 219),
}

CHART_COLORS = {
    "Happy/Excited":  "#F1C40F",
    "Angry/Stressed": "#E74C3C",
    "Calm/Relaxed":   "#2ECC71",
    "Sad/Bored":      "#3498DB",
}

NORMAL_RANGES = {
    "Mean RR (ms)":     ("600 - 1200 ms",  600,  1200),
    "SDNN (ms)":        ("20 - 150 ms",      0,   150),
    "RMSSD (ms)":       ("15 - 100 ms",      0,   100),
    "Heart Rate (bpm)": ("60 - 100 bpm",    40,   120),
    "pNN50 (%)":        ("5 - 40 %",         0,    50),
    "RR Range (ms)":    ("50 - 400 ms",      0,   500),
}


# ── Chart Generators ─────────────────────────────────────────

def generate_ecg_chart(signal_preview: list, emotion: str) -> str:
    color = CHART_COLORS.get(emotion, "#2980b9")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(signal_preview, color=color, linewidth=0.8)
    ax.set_title("ECG Signal Preview (first 2 seconds)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Sample (256 Hz)")
    ax.set_ylabel("Amplitude (normalised)")
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = "chart_ecg.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_prob_chart(all_probs: dict) -> str:
    emotions = list(all_probs.keys())
    probs    = list(all_probs.values())
    colors   = [CHART_COLORS[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(emotions, probs, color=colors, edgecolor='white', height=0.5)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{prob}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Emotion Probability Distribution", fontsize=11, fontweight='bold')
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = "chart_probs.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_hrv_chart(hrv_features: dict) -> str:
    labels, values, norm_values = [], [], []
    for k, v in hrv_features.items():
        if k in NORMAL_RANGES:
            _, lo, hi = NORMAL_RANGES[k]
            norm = max(0, min(100, (v - lo) / (hi - lo) * 100))
            labels.append(k)
            values.append(v)
            norm_values.append(norm)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(labels, norm_values, color='#2980b9', alpha=0.75, edgecolor='white')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel("Normalised Value (%)")
    ax.set_title("HRV Feature Overview", fontsize=11, fontweight='bold')
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=20, ha='right', fontsize=8)
    plt.tight_layout()
    path = "chart_hrv.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── PDF Class ────────────────────────────────────────────────

class ECGReport(FPDF):
    def header(self):
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 16)
        self.cell(0, 14, "ECG Emotion Recognition Report", fill=True, ln=True, align="C")
        self.set_font("Arial", "", 9)
        self.cell(0, 7,
                  f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
                  ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10,
                  f"Page {self.page_no()} | ECG Emotion Detector - For research use only",
                  align="C")

    def section_title(self, title: str):
        self.set_fill_color(236, 240, 241)
        self.set_font("Arial", "B", 12)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, f"  {title}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)


# ── Main Report Generator ────────────────────────────────────

def generate_report(result: dict) -> str:
    # Generate chart images
    ecg_chart  = generate_ecg_chart(result.get("signal_preview", []), result["emotion"])
    prob_chart = generate_prob_chart(result["all_probs"])
    hrv_chart  = generate_hrv_chart(result["hrv_features"])

    pdf = ECGReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Section 1: Detected Emotion ──
    pdf.section_title("1. Detected Emotion")

    r, g, b = BAR_COLORS[result["emotion"]]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 22)
    pdf.cell(0, 18, result["emotion"], fill=True, ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Confidence Score: {result['confidence']}%", ln=True, align="C")

    pdf.set_font("Arial", "I", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, result["description"], ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    pdf.set_font("Arial", "", 10)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(95, 8,
             f"  Original Sampling Rate: {result.get('sampling_rate', 'N/A')} Hz",
             border=1, fill=True)
    pdf.cell(95, 8,
             f"  Signal Duration: {result.get('signal_length_sec', 'N/A')} seconds",
             border=1, fill=True, ln=True)
    pdf.ln(8)

    # ── Section 2: ECG Signal Chart ──
    pdf.section_title("2. ECG Signal Preview")
    pdf.image(ecg_chart, x=10, w=190)
    pdf.ln(6)

    # ── Section 3: Emotion Probabilities ──
    pdf.section_title("3. Emotion Probability Distribution")
    pdf.image(prob_chart, x=15, w=180)
    pdf.ln(4)

    # Probability table
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(120, 8, "  Emotion",    fill=True, border=1)
    pdf.cell(70,  8, "  Confidence", fill=True, border=1, ln=True)
    pdf.set_text_color(0, 0, 0)

    for emotion, prob in result["all_probs"].items():
        is_top = emotion == result["emotion"]
        r, g, b = BAR_COLORS[emotion]
        pdf.set_fill_color(r, g, b) if is_top else pdf.set_fill_color(245, 245, 245)
        pdf.set_text_color(255, 255, 255) if is_top else pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B" if is_top else "", 10)
        label = f"  {'>> ' if is_top else ''}{emotion}"
        pdf.cell(120, 8, label,        border=1, fill=True)
        pdf.cell(70,  8, f"  {prob}%", border=1, fill=True, ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ── Section 4: HRV Analysis ──
    pdf.section_title("4. Heart Rate Variability (HRV) Analysis")
    pdf.image(hrv_chart, x=10, w=190)
    pdf.ln(4)

    # HRV table
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 8, "  HRV Feature",  fill=True, border=1)
    pdf.cell(50, 8, "  Value",        fill=True, border=1)
    pdf.cell(50, 8, "  Normal Range", fill=True, border=1, ln=True)
    pdf.set_text_color(0, 0, 0)

    for i, (k, v) in enumerate(result["hrv_features"].items()):
        fill_color = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.set_font("Arial", "", 10)
        normal_str = NORMAL_RANGES.get(k, ("N/A", 0, 0))[0]
        pdf.cell(90, 8, f"  {k}",          border=1, fill=True)
        pdf.cell(50, 8, f"  {v}",          border=1, fill=True)
        pdf.cell(50, 8, f"  {normal_str}", border=1, fill=True, ln=True)
    pdf.ln(8)

    # ── Section 5: Interpretation ──
    pdf.section_title("5. Clinical Interpretation Notes")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(60, 60, 60)

    interpretations = {
        "Happy/Excited": (
            "The detected emotional state suggests positive valence with high arousal. "
            "HRV patterns associated with this state typically show moderate RMSSD values "
            "and elevated heart rate. The sympathetic nervous system may be more active."
        ),
        "Angry/Stressed": (
            "The signal indicates high arousal with negative valence. Stress-related HRV "
            "patterns commonly show reduced SDNN and pNN50, with elevated heart rate. "
            "Sympathetic dominance is likely present."
        ),
        "Calm/Relaxed": (
            "The detected state suggests positive valence with low arousal. This is "
            "associated with parasympathetic dominance, typically showing higher RMSSD "
            "and pNN50 values and a lower, stable heart rate."
        ),
        "Sad/Bored": (
            "Low arousal with negative valence is detected. This state may be associated "
            "with reduced HRV overall and lower heart rate. Parasympathetic activity "
            "may be present but with reduced emotional engagement."
        ),
    }
    pdf.multi_cell(0, 6, interpretations.get(result["emotion"], ""))
    pdf.ln(6)

    # Disclaimer
    pdf.set_font("Arial", "I", 8)
    pdf.set_fill_color(255, 243, 205)
    pdf.set_text_color(100, 80, 0)
    pdf.multi_cell(0, 5,
        "Disclaimer: This report is generated by an AI model trained on the DREAMER "
        "research dataset. It is intended for research and educational purposes only. "
        "Results should not be used for clinical diagnosis or medical decision-making. "
        "Please consult a qualified healthcare professional for medical advice.",
        fill=True)

    # Save PDF
    output_path = "report_output.pdf"
    pdf.output(output_path)

    # Clean up chart images
    for chart in [ecg_chart, prob_chart, hrv_chart]:
        if os.path.exists(chart):
            os.remove(chart)

    return output_path
