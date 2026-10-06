"""
src/features/engineer.py
-------------------------
Rolling-window feature engineering for OSA prediction.
Operates on the 1-minute signal grid produced by fetch_physionet.py.

Features computed (at each minute t):
  Rolling windows: 5, 15, 30, 60 min
    - HR: mean, trend (linear slope), std
    - HRV RMSSD: mean, trend
    - SpO2: mean, trend, dip count (drops > 3% within window)
  Recency:
    - minutes_since_last_apnea: minutes since last annotated apnea minute
                                (only available for training, not live inference)
  Missingness:
    - hr_missing_frac, hrv_missing_frac, spo2_missing_frac per window
  Target variable:
    - event_within_30min: 1 if any apnea annotation in [t+1, t+30], else 0
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
WINDOWS = [5, 15, 30, 60]


# ─── Core rolling feature functions ──────────────────────────────────────────

def linear_trend(series: np.ndarray) -> float:
    """Slope of linear regression over the series index."""
    n = len(series)
    valid = ~np.isnan(series)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(n)[valid]
    y = series[valid]
    slope, *_ = stats.linregress(x, y)
    return float(slope)


def spo2_dip_count(series: np.ndarray, threshold: float = 3.0) -> float:
    """Count of SpO2 drops >= threshold% within the series."""
    valid = series[~np.isnan(series)]
    if len(valid) < 2:
        return 0.0
    baseline = np.nanpercentile(valid, 90)  # 90th pct as "baseline"
    dips = np.sum((baseline - valid) >= threshold)
    return float(dips)


def missing_fraction(series: np.ndarray) -> float:
    return float(np.sum(np.isnan(series)) / max(1, len(series)))


def rolling_features_for_window(
    df: pd.DataFrame,
    t: int,
    window: int,
    prefix: str,
) -> dict:
    """Compute features for a rolling look-back window ending at minute t."""
    start = max(0, t - window + 1)
    seg = df.iloc[start: t + 1]

    hr = seg["hr_bpm"].values
    hrv = seg["hrv_rmssd"].values
    spo2 = seg["spo2"].values

    feats = {}
    w = str(window)

    # HR features
    feats[f"hr_mean_{w}m"] = float(np.nanmean(hr)) if not np.all(np.isnan(hr)) else np.nan
    feats[f"hr_std_{w}m"] = float(np.nanstd(hr)) if not np.all(np.isnan(hr)) else np.nan
    feats[f"hr_trend_{w}m"] = linear_trend(hr)
    feats[f"hr_missing_{w}m"] = missing_fraction(hr)

    # HRV features
    feats[f"hrv_mean_{w}m"] = float(np.nanmean(hrv)) if not np.all(np.isnan(hrv)) else np.nan
    feats[f"hrv_trend_{w}m"] = linear_trend(hrv)
    feats[f"hrv_missing_{w}m"] = missing_fraction(hrv)

    # SpO2 features
    feats[f"spo2_mean_{w}m"] = float(np.nanmean(spo2)) if not np.all(np.isnan(spo2)) else np.nan
    feats[f"spo2_trend_{w}m"] = linear_trend(spo2)
    feats[f"spo2_dips_{w}m"] = spo2_dip_count(spo2)
    feats[f"spo2_missing_{w}m"] = missing_fraction(spo2)

    # SpO2 min (most clinically significant)
    feats[f"spo2_min_{w}m"] = float(np.nanmin(spo2)) if not np.all(np.isnan(spo2)) else np.nan

    return feats


def engineer_features_for_subject(
    signal_df: pd.DataFrame,
    ehr_row: Optional[pd.Series] = None,
    prediction_horizon: int = 30,
) -> pd.DataFrame:
    """
    Build full feature matrix for one subject.
    
    Args:
        signal_df: Per-minute signal data (from fetch_physionet.py)
        ehr_row: Optional Series with EHR features (from generate_ehr.py)
        prediction_horizon: Minutes ahead to predict (default 30)
    
    Returns:
        DataFrame with one row per minute, all features + target label
    """
    df = signal_df.reset_index(drop=True).copy()
    n = len(df)

    rows = []
    for t in range(n):
        row_feats: dict = {
            "subject_id": df.loc[t, "subject_id"],
            "minute": int(df.loc[t, "minute"]),
            # Current-minute raw values (always include for live inference context)
            "hr_current": df.loc[t, "hr_bpm"],
            "hrv_current": df.loc[t, "hrv_rmssd"],
            "spo2_current": df.loc[t, "spo2"],
            "apnea_now": int(df.loc[t, "apnea_label"]),
        }

        # Rolling window features across all window sizes
        for w in WINDOWS:
            row_feats.update(rolling_features_for_window(df, t, w, prefix=""))

        # Minutes since last apnea (training-time feature, set to -1 for inference
        # before any events have occurred)
        past_apnea = df["apnea_label"].iloc[:t]
        last_apnea_indices = past_apnea[past_apnea == 1].index
        if len(last_apnea_indices) > 0:
            row_feats["minutes_since_last_apnea"] = t - int(last_apnea_indices[-1])
        else:
            row_feats["minutes_since_last_apnea"] = -1  # none yet tonight

        # Cumulative apnea count so far this night
        row_feats["apnea_count_so_far"] = int(df["apnea_label"].iloc[:t].sum())

        # EHR static features (if provided)
        if ehr_row is not None:
            for col in [
                "age", "bmi", "neck_circumference_cm",
                "hypertension", "sedative_use", "alcohol_use",
                "prior_ahi", "ess_score", "stopbang_score",
                "sex",  # will be encoded below
            ]:
                val = ehr_row.get(col, np.nan)
                if col == "sex":
                    row_feats["sex_male"] = 1 if val == "M" else 0
                else:
                    row_feats[col] = val

        # Target: any apnea in next `prediction_horizon` minutes
        future = df["apnea_label"].iloc[t + 1: t + 1 + prediction_horizon]
        row_feats["event_within_30min"] = int(future.sum() > 0)

        rows.append(row_feats)

    return pd.DataFrame(rows)


def engineer_all_subjects(
    subjects_df: pd.DataFrame,
    ehr_df: Optional[pd.DataFrame] = None,
    prediction_horizon: int = 30,
) -> pd.DataFrame:
    """Process all subjects and concatenate into one feature matrix."""
    all_frames = []

    for _, subj in subjects_df.iterrows():
        sid = subj["subject_id"]
        sig_path = PROCESSED_DIR / f"signals_{sid}.parquet"

        if not sig_path.exists():
            print(f"  Skipping {sid}: signal file not found")
            continue

        sig_df = pd.read_parquet(sig_path)

        ehr_row = None
        if ehr_df is not None:
            mask = ehr_df["subject_id"] == sid
            if mask.any():
                ehr_row = ehr_df[mask].iloc[0]

        feat_df = engineer_features_for_subject(
            sig_df, ehr_row, prediction_horizon
        )
        all_frames.append(feat_df)
        print(f"  OK {sid}: {len(feat_df)} rows, {feat_df['event_within_30min'].mean():.1%} positive rate")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered list of model feature columns (excludes metadata/target)."""
    exclude = {
        "subject_id", "minute", "apnea_now", "apnea_label",
        "event_within_30min",
    }
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    subjects_df = pd.read_parquet(PROCESSED_DIR / "subjects.parquet")
    ehr_df = pd.read_parquet(PROCESSED_DIR / "ehr_profiles.parquet")

    print(f"Engineering features for {len(subjects_df)} subjects...")
    features_df = engineer_all_subjects(subjects_df, ehr_df)

    out_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"\nOK Features saved to {out_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Positive rate (30-min horizon): {features_df['event_within_30min'].mean():.1%}")
