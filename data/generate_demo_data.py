"""
generate_demo_data.py
---------------------
Generates REALISTIC synthetic signal data with the EXACT same schema as
fetch_physionet.py output, for immediate demo/validation without network access.

This creates physiologically plausible HR/HRV/SpO2 time series with:
  - Realistic baseline values (HR ~60-80 bpm, HRV 20-50ms, SpO2 94-99%)
  - Authentic apnea-event patterns: HR acceleration + HRV suppression +
    SpO2 desaturation, clustered into multi-minute runs
  - Subject-level OSA severity variation (severe subjects have more/longer events)
  - Temporal autocorrelation (values don't jump randomly)

Outputs the SAME files as fetch_physionet.py:
  data/processed/subjects.parquet
  data/processed/signals_<id>.parquet (one per subject)

NOTE: This is clearly labeled as SYNTHETIC in all outputs.
Real PhysioNet data can be loaded on top at any time by running
fetch_physionet.py -- it will overwrite these files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# Subject definitions (mirror real PhysioNet record categories)
# a-series: severe OSA; b-series: borderline; c/x-series: control/mild
DEMO_SUBJECTS = [
    ("a01", "severe",   52, 480),
    ("a02", "severe",   45, 490),
    ("a03", "severe",   38, 470),
    ("a04", "moderate", 28, 460),
    ("a05", "moderate", 22, 480),
    ("b01", "mild",     12, 450),
    ("b02", "mild",      8, 460),
    ("c01", "normal",    3, 440),
    ("c02", "normal",    1, 450),
    ("x01", "severe",   60, 480),
]

SEV_PARAMS = {
    "normal":   dict(apnea_rate=0.03, event_len=(1,3),  gap_len=(15,40), hr_base=62, hrv_base=40, spo2_base=97.5, spo2_drop_max=3.0),
    "mild":     dict(apnea_rate=0.10, event_len=(1,4),  gap_len=(8,20),  hr_base=67, hrv_base=30, spo2_base=96.0, spo2_drop_max=5.0),
    "moderate": dict(apnea_rate=0.22, event_len=(2,6),  gap_len=(4,12),  hr_base=72, hrv_base=22, spo2_base=95.0, spo2_drop_max=7.0),
    "severe":   dict(apnea_rate=0.40, event_len=(3,10), gap_len=(2,6),   hr_base=78, hrv_base=15, spo2_base=93.0, spo2_drop_max=10.0),
}


def _smooth(arr, window=3):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def generate_night_signal(subject_id, severity, n_minutes):
    p = SEV_PARAMS[severity]
    labels = np.zeros(n_minutes, dtype=int)

    # Generate apnea label sequence (clustered runs)
    t = 0
    while t < n_minutes:
        if rng.random() < p["apnea_rate"]:
            elen = int(rng.integers(p["event_len"][0], p["event_len"][1] + 1))
            end = min(t + elen, n_minutes)
            labels[t:end] = 1
            t = end
        else:
            gap = int(rng.integers(p["gap_len"][0], p["gap_len"][1] + 1))
            t += gap

    # HR: base + slow drift + apnea arousal spike
    hr_base = p["hr_base"]
    hr = np.full(n_minutes, float(hr_base))
    hr += np.cumsum(rng.normal(0, 0.08, n_minutes)).clip(-8, 8)
    hr += rng.normal(0, 1.5, n_minutes)
    for i in range(n_minutes):
        if labels[i] == 1:
            hr[i] += rng.uniform(4, 12)
        elif i > 0 and labels[max(0, i-2):i].sum() > 0:
            hr[i] += rng.uniform(8, 18)
    hr = _smooth(hr, 2).clip(40, 130)

    # HRV RMSSD: suppressed during apnea
    hrv = np.full(n_minutes, float(p["hrv_base"]))
    hrv += np.cumsum(rng.normal(0, 0.05, n_minutes))
    hrv += rng.normal(0, 2.0, n_minutes)
    for i in range(n_minutes):
        if labels[i] == 1:
            hrv[i] -= rng.uniform(5, 15)
        elif i > 0 and labels[max(0, i-1)] == 1:
            hrv[i] -= rng.uniform(2, 8)
    hrv = _smooth(hrv, 3).clip(2, 120)

    # SpO2: drops 1-3 min AFTER apnea starts (physiological lag)
    spo2 = np.full(n_minutes, float(p["spo2_base"]))
    spo2 += np.cumsum(rng.normal(0, 0.02, n_minutes))
    spo2 += rng.normal(0, 0.3, n_minutes)
    for i in range(n_minutes):
        lb = labels[max(0, i-3):i+1]
        if lb.sum() >= 2:
            spo2[i] -= rng.uniform(2.0, p["spo2_drop_max"])
        elif lb.sum() == 1 and i > 0 and labels[i-1] == 1:
            spo2[i] -= rng.uniform(0.5, p["spo2_drop_max"] * 0.5)
    spo2 = _smooth(spo2, 2).clip(75, 100)

    hrv_sdnn = (hrv * rng.uniform(1.4, 1.8, n_minutes)).clip(3, 180)

    rows = []
    for i in range(n_minutes):
        rows.append({
            "subject_id":  subject_id,
            "minute":      i,
            "hr_bpm":      round(float(hr[i]), 2),
            "hrv_rmssd":   round(float(hrv[i]), 2),
            "hrv_sdnn":    round(float(hrv_sdnn[i]), 2),
            "n_beats":     int(round(hr[i])),
            "spo2":        round(float(spo2[i]), 2),
            "apnea_label": int(labels[i]),
        })
    return pd.DataFrame(rows)


def main():
    print("Generating realistic synthetic demo data (schema-compatible with PhysioNet output)...")
    print("NOTE: Synthetic data. Run fetch_physionet.py to replace with real data.\n")

    all_meta = []
    for subject_id, severity, ahi_approx, n_minutes in DEMO_SUBJECTS:
        sig_df = generate_night_signal(subject_id, severity, n_minutes)
        total_apnea = int(sig_df["apnea_label"].sum())
        meta = {
            "subject_id": subject_id,
            "n_minutes": n_minutes,
            "total_apnea_minutes": total_apnea,
            "ahi_approx": float(ahi_approx),
            "has_spo2": True,
            "fs": 100,
            "osa_severity": severity,
        }
        all_meta.append(meta)
        sig_df.to_parquet(PROCESSED_DIR / f"signals_{subject_id}.parquet", index=False)
        print(f"  {subject_id} ({severity}): {n_minutes} min, {total_apnea} apnea min ({total_apnea/n_minutes*100:.1f}%)")

    subjects_df = pd.DataFrame(all_meta)
    subjects_df.to_parquet(PROCESSED_DIR / "subjects.parquet", index=False)
    subjects_df.to_csv(PROCESSED_DIR / "subjects.csv", index=False)

    summary = {
        "data_sources": {
            "signals": "Synthetic demo data (physiologically plausible, same schema as PhysioNet)",
            "ehr_profiles": "Synthea-inspired synthetic profiles",
            "linkage_basis": "OSA severity band",
        },
        "n_subjects": len(subjects_df),
        "severity_distribution": subjects_df["osa_severity"].value_counts().to_dict(),
        "disclaimer": "THIS IS SYNTHETIC DEMO DATA. Run fetch_physionet.py to replace with real PhysioNet data.",
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{len(subjects_df)} subjects saved to {PROCESSED_DIR}/")
    print(f"Total minutes: {subjects_df['n_minutes'].sum()}")
    print(f"Total apnea minutes: {subjects_df['total_apnea_minutes'].sum()}")
    print("\nSeverity distribution:")
    print(subjects_df["osa_severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
