"""
fetch_physionet.py
------------------
Downloads PhysioNet Apnea-ECG records (70 subjects, 1-min expert annotations)
using the `wfdb` package and extracts HR, HRV, and SpO2 on a 1-minute grid
aligned to the clinical apnea labels.

Outputs:
  data/processed/subjects.parquet   – one row per subject with metadata
  data/processed/signals_<id>.parquet – one row per minute for each subject
  data/processed/events_<id>.parquet  – annotated apnea/no-apnea per minute

Usage:
  python data/fetch_physionet.py [--max_subjects N]
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

PHYSIONET_DB = "apnea-ecg"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# All 70 records in the Apnea-ECG database
ALL_RECORDS = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "b01", "b02", "b03", "b04", "b05",
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",
    "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
    "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
    "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
    "x31", "x32", "x33", "x34", "x35",
]


def detect_r_peaks(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """Simple pan-tompkins inspired R-peak detector (derivative + threshold)."""
    # Bandpass-like: differentiate and square
    diff = np.diff(ecg_signal.astype(float))
    squared = diff ** 2

    # Moving average (window ~150 ms)
    win = max(1, int(0.15 * fs))
    mav = np.convolve(squared, np.ones(win) / win, mode="same")

    # Adaptive threshold
    threshold = 0.3 * np.max(mav)
    above = mav > threshold

    # Find rising edges
    r_peaks = []
    in_peak = False
    peak_start = 0
    for i, v in enumerate(above):
        if v and not in_peak:
            in_peak = True
            peak_start = i
        elif not v and in_peak:
            in_peak = False
            # Refine to actual maximum in original ECG within this segment
            seg = ecg_signal[peak_start: i + 1]
            local_max = np.argmax(np.abs(seg)) + peak_start
            r_peaks.append(local_max)

    # Enforce refractory period: minimum 0.3 s between peaks
    refractory = int(0.3 * fs)
    filtered_peaks = []
    last = -refractory
    for p in r_peaks:
        if p - last >= refractory:
            filtered_peaks.append(p)
            last = p

    return np.array(filtered_peaks)


def compute_hrv_features(rr_intervals_ms: np.ndarray) -> dict:
    """Compute basic HRV metrics from RR intervals (in ms)."""
    if len(rr_intervals_ms) < 2:
        return {"hrv_rmssd": np.nan, "hrv_sdnn": np.nan}
    diff_rr = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals_ms)
    return {"hrv_rmssd": float(rmssd), "hrv_sdnn": float(sdnn)}


def extract_minute_features(
    ecg: np.ndarray, fs: float, minute_idx: int
) -> dict:
    """Extract HR/HRV features for a single 1-minute window of ECG."""
    start = int(minute_idx * 60 * fs)
    end = int((minute_idx + 1) * 60 * fs)
    segment = ecg[start:end]

    if len(segment) < fs * 10:  # skip very short segments
        return {"hr_bpm": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan, "n_beats": 0}

    r_peaks = detect_r_peaks(segment, fs)

    if len(r_peaks) < 2:
        return {"hr_bpm": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan, "n_beats": len(r_peaks)}

    # RR intervals in milliseconds
    rr_ms = np.diff(r_peaks) / fs * 1000.0

    # Filter physiologically plausible RR (300–2000 ms → 30–200 bpm)
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]

    if len(rr_ms) < 2:
        return {"hr_bpm": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan, "n_beats": len(r_peaks)}

    hr_bpm = 60000.0 / np.mean(rr_ms)
    hrv_feats = compute_hrv_features(rr_ms)

    return {
        "hr_bpm": float(hr_bpm),
        "n_beats": len(r_peaks),
        **hrv_feats,
    }


def load_apnea_annotations(record_name: str) -> pd.DataFrame:
    """
    Load per-minute apnea annotations from PhysioNet Apnea-ECG.
    Returns DataFrame with columns: minute, apnea_label (1=apnea, 0=normal).
    """
    try:
        ann = wfdb.rdann(
            record_name,
            extension="apn",
            pn_dir=PHYSIONET_DB,
        )
        # Symbols: 'A' = apnea minute, 'N' = normal minute
        labels = []
        for sym in ann.symbol:
            labels.append(1 if sym == "A" else 0)
        minutes = list(range(len(labels)))
        return pd.DataFrame({"minute": minutes, "apnea_label": labels})
    except Exception as e:
        print(f"  Warning: Could not load annotations for {record_name}: {e}")
        return pd.DataFrame(columns=["minute", "apnea_label"])


def process_record(record_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Download and process one Apnea-ECG record.
    Returns (signal_df, metadata_dict).
    """
    print(f"  Processing {record_name}...")

    # Load record from PhysioNet
    try:
        record = wfdb.rdrecord(record_name, pn_dir=PHYSIONET_DB)
    except Exception as e:
        print(f"  ERROR loading {record_name}: {e}")
        return pd.DataFrame(), {}

    fs = record.fs
    # ECG is always channel 0; SpO2 is channel 1 if present
    ecg = record.p_signal[:, 0]
    has_spo2 = record.n_sig > 1
    spo2_signal = record.p_signal[:, 1] if has_spo2 else None

    # Load expert annotations
    ann_df = load_apnea_annotations(record_name)
    n_minutes = len(ann_df) if len(ann_df) > 0 else int(len(ecg) / (fs * 60))

    rows = []
    for min_idx in range(n_minutes):
        feats = extract_minute_features(ecg, fs, min_idx)

        # SpO2: average of the minute if available
        spo2_val = np.nan
        if spo2_signal is not None:
            start = int(min_idx * 60 * fs)
            end = int((min_idx + 1) * 60 * fs)
            seg_spo2 = spo2_signal[start:end]
            valid = seg_spo2[(seg_spo2 > 50) & (seg_spo2 <= 100)]
            if len(valid) > 0:
                spo2_val = float(np.mean(valid))

        # Apnea label for this minute
        apnea_label = 0
        if len(ann_df) > min_idx:
            apnea_label = int(ann_df.loc[min_idx, "apnea_label"])

        rows.append({
            "subject_id": record_name,
            "minute": min_idx,
            "hr_bpm": feats["hr_bpm"],
            "hrv_rmssd": feats["hrv_rmssd"],
            "hrv_sdnn": feats["hrv_sdnn"],
            "n_beats": feats["n_beats"],
            "spo2": spo2_val,
            "apnea_label": apnea_label,
        })

    signal_df = pd.DataFrame(rows)

    # Metadata
    total_apnea_minutes = int(signal_df["apnea_label"].sum())
    # AHI = apnea events per hour of recording (the clinically standard definition)
    # Each annotated minute = one apnea event. Divide by recording hours.
    recording_hours = n_minutes / 60.0
    ahi_approx = round(total_apnea_minutes / recording_hours, 1) if recording_hours > 0 else 0.0
    meta = {
        "subject_id": record_name,
        "n_minutes": n_minutes,
        "total_apnea_minutes": total_apnea_minutes,
        "ahi_approx": ahi_approx,
        "has_spo2": has_spo2,
        "fs": fs,
    }

    return signal_df, meta


def main(max_subjects: int = 70):
    records = ALL_RECORDS[:max_subjects]
    all_meta = []
    failed = []

    print(f"Fetching {len(records)} records from PhysioNet Apnea-ECG database...")
    print("(This downloads ~50-100 MB; subsequent runs use local cache.)\n")

    for rec in records:
        sig_df, meta = process_record(rec)
        if sig_df.empty:
            failed.append(rec)
            continue

        # Save per-subject signal file
        out_path = PROCESSED_DIR / f"signals_{rec}.parquet"
        sig_df.to_parquet(out_path, index=False)
        all_meta.append(meta)
        print(f"    OK {rec}: {meta['n_minutes']} min, {meta['total_apnea_minutes']} apnea min")

    if failed:
        print(f"\nFailed records: {failed}")

    # Save subjects manifest
    subjects_df = pd.DataFrame(all_meta)
    subjects_df.to_parquet(PROCESSED_DIR / "subjects.parquet", index=False)
    subjects_df.to_csv(PROCESSED_DIR / "subjects.csv", index=False)

    print(f"\nOK Processed {len(all_meta)} subjects.")
    print(f"  Saved to: {PROCESSED_DIR}")
    print(f"  Total apnea minutes: {subjects_df['total_apnea_minutes'].sum()}")
    print(f"  Total minutes: {subjects_df['n_minutes'].sum()}")

    # Severity classification based on approximate AHI
    def severity(ahi):
        if ahi < 5:
            return "normal"
        elif ahi < 15:
            return "mild"
        elif ahi < 30:
            return "moderate"
        else:
            return "severe"

    subjects_df["osa_severity"] = subjects_df["ahi_approx"].apply(severity)
    subjects_df.to_parquet(PROCESSED_DIR / "subjects.parquet", index=False)
    subjects_df.to_csv(PROCESSED_DIR / "subjects.csv", index=False)

    print("\nSeverity distribution:")
    print(subjects_df["osa_severity"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_subjects", type=int, default=70,
                        help="Max number of subjects to download (default: all 70)")
    args = parser.parse_args()
    main(args.max_subjects)
