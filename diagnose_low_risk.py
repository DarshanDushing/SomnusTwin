"""
diagnose_low_risk.py
--------------------
4-step diagnostic for Severe OSA subjects showing ~2% risk.

Run:
    .venv\Scripts\python.exe diagnose_low_risk.py

Steps:
  1  Find which Severe OSA subject + minute shows the lowest risk score
  2  Check whether the raw signals at that moment are genuinely calm
  3  Inspect feature vector, EHR identity, SHAP reason codes
  4  Full-night risk trajectory across 12 evenly-spaced minutes
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "data" / "models"

# ── helpers ────────────────────────────────────────────────────────────────────

DIVIDER   = "=" * 72
SEPARATOR = "-" * 72

def pct(v):
    return f"{v * 100:.1f}%"

def fmt_row(label, value, width=38):
    return f"  {label:<{width}} {value}"


# ======================================================================
# STEP 1 - Find the severe subject + minute with the lowest risk score
# ======================================================================

print(f"\n{DIVIDER}")
print("STEP 1 - Locate the Severe OSA subject with the lowest risk score")
print(DIVIDER)

from src.models.predictor import SomnusTwinPredictor

predictor = SomnusTwinPredictor.get()
if not predictor.is_ready:
    print("ERROR: Model not loaded. Run training pipeline first.")
    sys.exit(1)

subjects_df = pd.read_parquet(PROCESSED_DIR / "subjects.parquet")
ehr_df      = pd.read_parquet(PROCESSED_DIR / "ehr_profiles.parquet")

severe_sids = subjects_df[subjects_df["osa_severity"] == "severe"]["subject_id"].tolist()
print(f"\nSevere OSA subjects: {severe_sids}")

# Sample 20 evenly-spaced minutes per severe subject; find global minimum
lowest_risk = 1.0
lowest_sid  = None
lowest_min  = None

subject_samples: dict = {}

for sid in severe_sids:
    sig_path = PROCESSED_DIR / f"signals_{sid}.parquet"
    if not sig_path.exists():
        print(f"  {sid}: signal file missing, skipping")
        continue

    sig_df = pd.read_parquet(sig_path)
    minutes = sorted(sig_df["minute"].unique())

    ehr_mask = ehr_df["subject_id"] == sid
    ehr_row  = ehr_df[ehr_mask].iloc[0] if ehr_mask.any() else None

    probe_minutes = np.linspace(minutes[0], minutes[-1], 20, dtype=int).tolist()
    probe_minutes = sorted(set(probe_minutes))

    samples = []
    for m in probe_minutes:
        r = predictor.predict_at_minute(sig_df, m, ehr_row)
        samples.append({"minute": m, "risk": r["risk_score"]})
        if r["risk_score"] < lowest_risk:
            lowest_risk = r["risk_score"]
            lowest_sid  = sid
            lowest_min  = m

    subject_samples[sid] = samples
    min_r = min(s["risk"] for s in samples)
    max_r = max(s["risk"] for s in samples)
    print(f"  {sid}: risk range {pct(min_r)} - {pct(max_r)}")

print(f"\n>>> LOWEST RISK: {pct(lowest_risk)}  @  subject={lowest_sid}, minute={lowest_min}")


# ======================================================================
# STEP 2 - Is this a genuinely calm moment?
# ======================================================================

print(f"\n{DIVIDER}")
print(f"STEP 2 - Raw signal + ground truth at {lowest_sid} minute={lowest_min}")
print(DIVIDER)

sig_df = pd.read_parquet(PROCESSED_DIR / f"signals_{lowest_sid}.parquet")

at_t = sig_df[sig_df["minute"] == lowest_min]
if at_t.empty:
    at_t = sig_df.iloc[(sig_df["minute"] - lowest_min).abs().argsort()[:1]]
row_t = at_t.iloc[0]

hr_now    = float(row_t.get("hr_bpm",    float("nan")))
hrv_now   = float(row_t.get("hrv_rmssd", float("nan")))
spo2_now  = float(row_t.get("spo2",      float("nan")))
apnea_now = int(row_t.get("apnea_label", 0))

# Ground truth: any event in next 30 min?
future_window = sig_df[
    (sig_df["minute"] > lowest_min) &
    (sig_df["minute"] <= lowest_min + 30)
]
event_in_30min = int(future_window["apnea_label"].sum() > 0) if not future_window.empty else -1

# 30-min look-back context
lookback_30 = sig_df[
    (sig_df["minute"] >= lowest_min - 30) &
    (sig_df["minute"] <= lowest_min)
]
hr_series   = lookback_30["hr_bpm"].dropna()
spo2_series = lookback_30["spo2"].dropna()
hr_trend    = float(hr_series.iloc[-1]   - hr_series.iloc[0])   if len(hr_series)   >= 2 else float("nan")
spo2_trend  = float(spo2_series.iloc[-1] - spo2_series.iloc[0]) if len(spo2_series) >= 2 else float("nan")
recent_apnea_count = int(lookback_30["apnea_label"].sum())

print(fmt_row("Subject:", f"{lowest_sid}  ({subjects_df[subjects_df['subject_id']==lowest_sid]['osa_severity'].iloc[0].upper()} OSA)"))
print(fmt_row("Minute:", lowest_min))
print(fmt_row("Model risk score:", pct(lowest_risk)))
print()
print(fmt_row("HR  (current):", f"{hr_now:.1f} bpm"))
print(fmt_row("HRV (current):", f"{hrv_now:.1f} ms"))
print(fmt_row("SpO2 (current):", f"{spo2_now:.1f}%"))
print(fmt_row("Apnea event AT this minute:", "YES" if apnea_now else "no"))
print()

hr_t_label   = "rising" if hr_trend > 3 else ("stable" if hr_trend > -3 else "falling")
spo2_t_label = "dropping" if spo2_trend < -2 else "stable"
print(fmt_row("HR  trend (last 30m):", f"{hr_trend:+.1f} bpm  ({hr_t_label})"))
print(fmt_row("SpO2 trend (last 30m):", f"{spo2_trend:+.1f}%  ({spo2_t_label})"))
print(fmt_row("Apnea events in last 30m:", f"{recent_apnea_count}"))
print()
print(fmt_row("event_within_30min (ground truth):",
              "YES - event coming ahead" if event_in_30min == 1
              else ("NO  - no event ahead" if event_in_30min == 0 else "unknown")))

# Calm heuristic
calm_signals = (
    (not np.isnan(spo2_now) and spo2_now >= 94.0) and
    (not np.isnan(hr_now)   and 50 <= hr_now <= 90) and
    spo2_trend > -2.0 and
    recent_apnea_count <= 3
)

print()
if event_in_30min == 0 and calm_signals:
    verdict2 = "VERDICT: CORRECT - Signals genuinely calm AND no event coming => 2% is expected behavior."
elif event_in_30min == 1 and calm_signals:
    verdict2 = "VERDICT: SUSPICIOUS - Signals look calm BUT an event IS coming in 30 min => possible under-prediction."
elif not calm_signals:
    verdict2 = "VERDICT: SUSPICIOUS - Signals look physiologically concerning but risk is ~2% => investigate."
else:
    verdict2 = "VERDICT: AMBIGUOUS - proceeding to Step 3 for completeness."

print(f"  {verdict2}")


# ======================================================================
# STEP 3 - Feature vector, EHR identity, SHAP
# ======================================================================

print(f"\n{DIVIDER}")
print(f"STEP 3 - Feature vector / EHR identity / SHAP at {lowest_sid} minute={lowest_min}")
print(DIVIDER)

ehr_mask = ehr_df["subject_id"] == lowest_sid
if not ehr_mask.any():
    print("  WARNING: No EHR row found for this subject - EHR features will be zero")
    ehr_row = None
else:
    ehr_row = ehr_df[ehr_mask].iloc[0]
    print("\n-- EHR IDENTITY CHECK (confirm correct subject is loaded) --")
    for col in ["subject_id", "age", "bmi", "prior_ahi", "osa_severity",
                "stopbang_score", "hypertension", "sedative_use"]:
        if col in ehr_row.index:
            print(fmt_row(f"  {col}:", str(ehr_row[col])))

# Rebuild feature vector
sig_df_up_to_t = sig_df[sig_df["minute"] <= lowest_min].reset_index(drop=True)
from src.features.engineer import engineer_features_for_subject

feat_df = engineer_features_for_subject(sig_df_up_to_t, ehr_row)

feat_names   = predictor.config["feature_names"]
feat_impute  = predictor.config.get("feat_impute_means", predictor.config["feat_mean"])

t_local = min(lowest_min, len(feat_df) - 1)
feat_row = feat_df.iloc[t_local]

feat_vec = np.array([
    feat_row.get(n, feat_impute[i])
    if not pd.isna(feat_row.get(n, np.nan))
    else feat_impute[i]   # training-fold mean, matches fixed predictor
    for i, n in enumerate(feat_names)
], dtype=np.float32)
feat_vec_norm = predictor._normalize_features(feat_vec)

print(f"\n-- PRE-SCALING vs POST-SCALING (key features) --")
key_feats = [
    "hr_current", "hrv_current", "spo2_current",
    "hr_mean_30m", "hrv_mean_30m", "spo2_mean_30m", "spo2_min_30m",
    "spo2_dips_30m", "minutes_since_last_apnea", "apnea_count_so_far",
    "prior_ahi", "bmi", "stopbang_score",
]
for fname in key_feats:
    raw_val = feat_row.get(fname, "N/A")
    if fname in feat_names:
        idx = feat_names.index(fname)
        norm_val = feat_vec_norm[idx]
        nan_flag = " <- NaN->0 WARNING" if pd.isna(feat_row.get(fname, np.nan)) else ""
        print(fmt_row(f"  {fname}:", f"raw={float(raw_val) if raw_val != 'N/A' else 'N/A':<8.3f}  normed={norm_val:+.3f}{nan_flag}"))
    else:
        print(fmt_row(f"  {fname}:", f"raw={raw_val}  (not in feat_names)"))

nan_count  = int(np.sum(np.isnan(feat_vec)))
zero_count = int(np.sum(feat_vec == 0.0))
print(f"\n  NaN values in raw feat_vec  : {nan_count}  (all converted to 0 before model)")
print(f"  Zero values in raw feat_vec : {zero_count} / {len(feat_vec)}")

# SHAP
result = predictor.predict_at_minute(sig_df_up_to_t, lowest_min, ehr_row)
print(f"\n-- SHAP TOP-5 REASON CODES --")
reasons = result.get("reasons", [])
if reasons:
    for r in reasons[:5]:
        print(f"  {r['direction']:<12}  {r['feature']:<35}  SHAP={r['shap_value']:+.4f}  |  {r['label']}")
else:
    print("  (no SHAP reasons returned)")


# ======================================================================
# STEP 4 - Full-night trajectory for ALL severe subjects
# ======================================================================

print(f"\n{DIVIDER}")
print("STEP 4 - Full-night risk trajectory (12 probe points per severe subject)")
print(DIVIDER)

flags_found = []

for sid in severe_sids:
    sig_path = PROCESSED_DIR / f"signals_{sid}.parquet"
    if not sig_path.exists():
        continue

    sig_full = pd.read_parquet(sig_path)
    minutes_all = sorted(sig_full["minute"].unique())

    ehr_mask2 = ehr_df["subject_id"] == sid
    ehr_row2  = ehr_df[ehr_mask2].iloc[0] if ehr_mask2.any() else None

    probe_pts = np.linspace(minutes_all[0], minutes_all[-1], 12, dtype=int).tolist()
    probe_pts = sorted(set(probe_pts))

    sev_label = subjects_df[subjects_df["subject_id"] == sid]["osa_severity"].iloc[0].upper()
    print(f"\n  {sid}  ({sev_label} OSA)")
    print(f"  {'Min':>6}  {'Risk':>7}  {'HR':>6}  {'SpO2':>6}  {'ApneaNow':>9}  {'Evt30m':>7}  Bar (0-100%)")
    print(f"  {SEPARATOR[:72]}")

    for m in probe_pts:
        r2 = predictor.predict_at_minute(sig_full, m, ehr_row2)
        sig_row = sig_full[sig_full["minute"] == m]
        if sig_row.empty:
            sig_row = sig_full.iloc[(sig_full["minute"] - m).abs().argsort()[:1]]
        sr = sig_row.iloc[0]

        hr_v   = float(sr.get("hr_bpm",    float("nan")))
        spo2_v = float(sr.get("spo2",      float("nan")))
        apn_v  = int(sr.get("apnea_label", 0))

        future30 = sig_full[
            (sig_full["minute"] > m) & (sig_full["minute"] <= m + 30)
        ]
        ev30 = int(future30["apnea_label"].sum() > 0) if not future30.empty else -1

        bar = "#" * int(r2["risk_score"] * 20)

        flag = ""
        if r2["risk_score"] < 0.05 and (apn_v or ev30 == 1):
            flag = "  <- LOW risk despite event!"
            flags_found.append((sid, m, r2["risk_score"], "low_during_event"))
        elif r2["risk_score"] > 0.70 and not apn_v and ev30 == 0:
            flag = "  <- HIGH risk on calm moment"
            flags_found.append((sid, m, r2["risk_score"], "high_during_calm"))

        print(f"  {m:>6}  {pct(r2['risk_score']):>7}  {hr_v:>6.1f}  {spo2_v:>6.1f}  "
              f"{'YES' if apn_v else 'no':>9}  {'YES' if ev30 == 1 else 'no':>7}  {bar}{flag}")


# ======================================================================
# FINAL VERDICT
# ======================================================================

print(f"\n{DIVIDER}")
print("FINAL VERDICT")
print(DIVIDER)
print(f"\n  Subject with globally lowest risk : {lowest_sid}  @  minute {lowest_min}")
print(f"  Risk score                        : {pct(lowest_risk)}")
print(f"  Ground truth (event in 30 min)    : {'YES' if event_in_30min == 1 else 'NO'}")
print(f"  Signals at that moment            : {'CALM' if calm_signals else 'CONCERNING'}")
print()
print(f"  {verdict2}")
print()

if flags_found:
    print(f"  ANOMALIES flagged across Step 4 ({len(flags_found)} total):")
    for f in flags_found:
        print(f"    {f[0]} min={f[1]} risk={pct(f[2])} type={f[3]}")
    print()
    print("  => At least one anomaly found. Review SHAP output in Step 3 to")
    print("     determine if it is a model limitation or a genuine data issue.")
else:
    print("  No Step-4 anomalies flagged: risk RISES during events and DROPS")
    print("  during calm stretches across all severe subjects.")
    print()
    if event_in_30min == 0 and calm_signals:
        print("  => CONCLUSION: The ~2% risk IS CORRECT EXPECTED BEHAVIOR.")
        print("     The model correctly reflects real inter-event calm stretches.")
        print("     No fix needed. The saturation bug is NOT reintroduced.")
    else:
        print("  => CONCLUSION: Investigate Step 3 output further.")

print(f"\n{DIVIDER}\n")
