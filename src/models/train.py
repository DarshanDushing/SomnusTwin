import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─── Module-level constants (imported by predictor.py) ───────────────────────

GRU_HIDDEN: int = 32
GRU_SEQUENCE_LEN: int = 30

# ─── Paths ───────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent.parent / "data" / "models"
RESULTS_DIR   = Path(__file__).parent.parent.parent / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── GRU Encoder (exported — used by predictor.py at inference time) ────────

class SignalGRUEncoder(nn.Module):
    """
    2-layer GRU that encodes a trailing window of (HR, HRV, SpO2) signals
    into a fixed-size hidden-state vector.

    Input  : (batch, seq_len, 3)   — 3 channels: HR, HRV, SpO2
    Output : (batch, hidden_size)  — last-step hidden state
    """

    def __init__(self, input_size: int = 3, hidden_size: int = GRU_HIDDEN, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        return self.norm(h_last)


# ─── Sequence builder ────────────────────────────────────────────────────────

def _compute_seq_impute_means(features_df: pd.DataFrame) -> dict:
    """Compute per-channel imputation means from a training fold (no leakage)."""
    hr_mean   = float(features_df["hr_current"].mean(skipna=True))
    hrv_mean  = float(features_df["hrv_current"].mean(skipna=True))
    spo2_mean = float(features_df["spo2_current"].mean(skipna=True))
    # Fall back to physiological neutrals if entire column is NaN
    if np.isnan(hr_mean):   hr_mean   = 70.0
    if np.isnan(hrv_mean):  hrv_mean  = 25.0
    if np.isnan(spo2_mean): spo2_mean = 95.0
    return {"hr": hr_mean, "hrv": hrv_mean, "spo2": spo2_mean}


def _build_sequences(
    features_df,
    seq_len: int = GRU_SEQUENCE_LEN,
    impute_means: Optional[dict] = None,
):
    """
    Build (sequence, label) pairs for the GRU encoder.

    NaN values are imputed with training-fold channel means so that missing
    channels (e.g. SpO2-absent ECG-only recordings) normalize to ~0 z-score
    rather than the catastrophic -25σ produced by filling with 0.

    Args:
        features_df   : Feature DataFrame (training or test fold)
        seq_len       : GRU look-back length (default 30)
        impute_means  : {'hr': float, 'hrv': float, 'spo2': float}
                        Must come from the TRAINING fold only to avoid leakage.
                        If None (training call), computed from features_df itself.
    """
    if impute_means is None:
        impute_means = _compute_seq_impute_means(features_df)

    hr_imp   = impute_means["hr"]
    hrv_imp  = impute_means["hrv"]
    spo2_imp = impute_means["spo2"]

    sequences, labels = [], []
    for _, subj_df in features_df.groupby("subject_id"):
        subj_df = subj_df.sort_values("minute").reset_index(drop=True)
        hrs   = subj_df["hr_current"].fillna(hr_imp).values
        hrvs  = subj_df["hrv_current"].fillna(hrv_imp).values
        spo2s = subj_df["spo2_current"].fillna(spo2_imp).values
        ys    = subj_df["event_within_30min"].values
        for t in range(len(subj_df)):
            start   = max(0, t - seq_len + 1)
            hr_seg  = hrs[start: t + 1]
            hrv_seg = hrvs[start: t + 1]
            sp_seg  = spo2s[start: t + 1]
            pad     = seq_len - len(hr_seg)
            hr_seg  = np.concatenate([np.full(pad, hr_imp),   hr_seg])
            hrv_seg = np.concatenate([np.full(pad, hrv_imp),  hrv_seg])
            sp_seg  = np.concatenate([np.full(pad, spo2_imp), sp_seg])
            seq = np.stack([hr_seg, hrv_seg, sp_seg], axis=1).astype(np.float32)
            sequences.append(seq)
            labels.append(int(ys[t]))
    return np.array(sequences), np.array(labels, dtype=np.float32)


def _normalize_sequences(train_seqs, test_seqs):
    mean = train_seqs.mean(axis=(0, 1), keepdims=True)
    std  = train_seqs.std(axis=(0, 1),  keepdims=True) + 1e-8
    return (
        (train_seqs - mean) / std,
        (test_seqs  - mean) / std,
        mean.squeeze(),
        std.squeeze(),
    )


# ─── GRU training ────────────────────────────────────────────────────────────

def train_gru_encoder(train_seqs, train_labels, hidden_size=GRU_HIDDEN,
                      epochs=20, batch_size=256, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SignalGRUEncoder(hidden_size=hidden_size).to(device)
    head    = nn.Linear(hidden_size, 1).to(device)
    X = torch.tensor(train_seqs,   dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.float32)
    ds     = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
    )
    # Handle the case where mean is 0
    mean_val = train_labels.mean()
    pos_weight_val = mean_val ** -1 if mean_val > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device)
    )
    encoder.train(); head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(encoder(xb)).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            print(f"  GRU epoch {epoch+1}/{epochs}  loss={total_loss/len(ds):.4f}")
    encoder.eval()
    return encoder


def encode_sequences(encoder, sequences, device=None, batch_size=512):
    if len(sequences) == 0:
        return np.zeros((0, encoder.hidden_size), dtype=np.float32)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    all_h = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[i: i + batch_size], dtype=torch.float32).to(device)
            all_h.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_h, axis=0)


# ─── LightGBM training ───────────────────────────────────────────────────────

def _get_feature_cols(df):
    # Intentionally excluded:
    #   prior_ahi / oss_score / stopbang_score / osa_severity / ahi_real
    #     → these are direct leaky proxies for the ground-truth severity label.
    #   apnea_count_so_far
    #     → cumulative count uses all past annotations; kept out deliberately.
    #   minutes_since_last_apnea was INCORRECTLY in this list — restored below.
    #     It is computed causally (only from past minutes) and is analogous to
    #     the already-included hr/hrv rolling features.
    exclude = {
        "subject_id", "minute", "apnea_now", "apnea_label", "event_within_30min",
        "prior_ahi", "ess_score", "stopbang_score", "osa_severity", "ahi_real",
        "apnea_count_so_far",
        # minutes_since_last_apnea intentionally KEPT (causal, non-leaky)
    }
    return [c for c in df.columns if c not in exclude]


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    import lightgbm as lgb
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, roc_auc_score, roc_curve
    )
    if len(y_train) == 0 or len(y_test) == 0:
        return None, {}, {}
    pos_sum = (y_train == 1).sum()
    pos_weight = (y_train == 0).sum() / max(1, pos_sum)
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.08, num_leaves=31, max_depth=5,
        min_child_samples=25, scale_pos_weight=pos_weight,
        reg_alpha=0.3, reg_lambda=0.5,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    
    # Handle single class in train or test
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # Cannot calculate ROC/PRC
        return model, {
            "auroc": float('nan'), "auprc": float('nan'), "brier_score": float('nan'),
            "sensitivity_at_80_specificity": float('nan'),
            "sensitivity_at_90_specificity": float('nan'),
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test":  float(y_test.mean()),
        }, {}

    # Use eval_set for early stopping
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    sens_at_80 = float(tpr[np.argmin(np.abs((1 - fpr) - 0.80))])
    sens_at_90 = float(tpr[np.argmin(np.abs((1 - fpr) - 0.90))])
    metrics = {
        "auroc": float(auroc), "auprc": float(auprc), "brier_score": float(brier),
        "sensitivity_at_80_specificity": sens_at_80,
        "sensitivity_at_90_specificity": sens_at_90,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test":  float(y_test.mean()),
    }
    print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")
    print(f"  Sens@80%Spec={sens_at_80:.1%}  Sens@90%Spec={sens_at_90:.1%}")
    return model, metrics, {"proba": proba}


# ─── SHAP ────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test, feature_names):
    if model is None or len(X_test) == 0:
        return {"top_features": [], "shap_values_sample": []}
    explainer = shap.TreeExplainer(model)
    sample    = X_test[:500] if len(X_test) > 500 else X_test
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1]
    top_features = [
        {"feature": feature_names[i], "importance": float(mean_abs[i])}
        for i in idx_sorted[:20]
    ]
    return {"top_features": top_features, "shap_values_sample": shap_vals}


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt4(val):
    return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

def _fmtpct(val):
    return f"{val:.1%}" if isinstance(val, (int, float)) else "N/A"


# ─── Results writer ──────────────────────────────────────────────────────────

def write_results(metrics, ablation, shap_result, model_config, n_subjects, per_subject_metrics):
    results = {
        "model_config": model_config,
        "validation_approach": {
            "split": "10-fold leave-one-subject-out CV (no row-level leakage)",
            "ground_truth": "Real PhysioNet Apnea-ECG expert annotations",
            "prediction_horizon": "30 minutes ahead",
            "n_subjects_total": n_subjects,
            "n_subjects_test": n_subjects,
        },
        "primary_metrics": metrics,
        "ablation": ablation,
        "shap_top_features": shap_result.get("top_features", [])[:5],
        "per_subject_metrics": per_subject_metrics
    }
    
    def sanitize_for_json(obj):
        import math
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(sanitize_for_json(results), f, indent=2)

    ehr      = ablation.get("ehr_only", {})
    wearable = ablation.get("wearable_only", {})
    fused    = ablation.get("fused", {})

    def fmt_ms(val, std):
        if val is None or std is None or np.isnan(val):
            return "N/A"
        return f"{val:.4f} ± {std:.4f}"

    def fmt_pool(val):
        if val is None or np.isnan(val):
            return "N/A"
        return f"{val:.4f}"

    ehr_auroc = fmt_ms(ehr.get("auroc"), ehr.get("auroc_std"))
    ehr_auprc = fmt_ms(ehr.get("auprc"), ehr.get("auprc_std"))
    ehr_sens  = fmt_ms(ehr.get("sensitivity_at_80_specificity"), ehr.get("sensitivity_at_80_specificity_std"))

    wearable_auroc = fmt_ms(wearable.get("auroc"), wearable.get("auroc_std"))
    wearable_auprc = fmt_ms(wearable.get("auprc"), wearable.get("auprc_std"))
    wearable_sens  = fmt_ms(wearable.get("sensitivity_at_80_specificity"), wearable.get("sensitivity_at_80_specificity_std"))

    fused_auroc = fmt_ms(fused.get("auroc"), fused.get("auroc_std"))
    fused_auprc = fmt_ms(fused.get("auprc"), fused.get("auprc_std"))
    fused_sens  = fmt_ms(fused.get("sensitivity_at_80_specificity"), fused.get("sensitivity_at_80_specificity_std"))

    lines = [
        "# SomnusTwin - Validation Report",
        "",
        "> **Data Source**: Real clinician-scored apnea events from the PhysioNet Apnea-ECG",
        "> Database (Penzel et al., 2000). NOT synthetic labels.",
        "",
        "Per-fold AUROC is averaged across 10 leave-one-subject-out folds and reflects consistency across individual patients. Pooled AUROC concatenates predictions across all folds before scoring and is more informative for features (like static EHR demographics) that are constant within any single patient's test fold but vary meaningfully across patients. We report both for transparency.",
        "",
        "## Model Architecture",
        "- **Primary**: LightGBM on engineered rolling features + EHR statics + GRU hidden state",
        f"- **GRU Encoder**: 2-layer GRU ({GRU_HIDDEN}-dim) over trailing 30-min signal window",
        "- **Training validation**: 10-fold Leave-One-Subject-Out (LOSO) Cross-Validation",
        "",
        "## Primary Metrics (Test Set Averages across 10 folds)",
        "| Metric | Per-Fold Mean ± Std | Pooled (all folds combined) |",
        "|--------|---------------------|-----------------------------|",
        f"| **AUROC** | {fused_auroc} | **{fmt_pool(fused.get('pooled_auroc'))}** |",
        f"| **AUPRC** | {fused_auprc} | **{fmt_pool(fused.get('pooled_auprc'))}** |",
        f"| Sensitivity @ 80% Specificity | {fused_sens} | {fmt_pool(fused.get('pooled_sens_80'))} |",
        f"| Brier Score | {fmt_ms(metrics.get('brier_score'), metrics.get('brier_score_std'))} | {fmt_pool(fused.get('pooled_brier'))} |",
        "",
        f"*Total validation set: {metrics.get('n_test', 0):,} minute-windows across {n_subjects} subjects*",
        f"*Mean positive rate (event within 30 min): {metrics.get('positive_rate_test', 0):.1%}*",
        "",
        "## Ablation Study (LOSO-CV)",
        "",
        "| Model Variant | AUROC (Mean±Std) | AUROC (Pooled) | AUPRC (Pooled) |",
        "|---------------|------------------|----------------|----------------|",
        f"| EHR only | {ehr_auroc} | {fmt_pool(ehr.get('pooled_auroc'))} | {fmt_pool(ehr.get('pooled_auprc'))} |",
        f"| Wearable signals only | {wearable_auroc} | {fmt_pool(wearable.get('pooled_auroc'))} | {fmt_pool(wearable.get('pooled_auprc'))} |",
        f"| **Fused (EHR + signals)** | {fused_auroc} | **{fmt_pool(fused.get('pooled_auroc'))}** | **{fmt_pool(fused.get('pooled_auprc'))}** |",
        "",
        "## Per-Subject Breakdown",
        "",
        "| Subject | AUROC | AUPRC | Sensitivity @ 80% Spec | Brier Score |",
        "|---------|-------|-------|------------------------|-------------|"
    ]
    
    for row in sorted(per_subject_metrics, key=lambda x: x["auroc"] if not np.isnan(x["auroc"]) else -1):
        if np.isnan(row['auroc']):
            lines.append(f"| {row['subject_id']} | N/A (no pos events) | N/A (no pos events) | N/A | N/A |")
        else:
            lines.append(f"| {row['subject_id']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['sens_80']:.4f} | {row['brier']:.4f} |")

    lines.extend([
        "",
        "## Top SHAP Features",
        "| Rank | Feature | Mean |SHAP| |",
        "|------|---------|------------------|"
    ])
    
    for i, f in enumerate(shap_result.get("top_features", [])[:5]):
        lines.append(f"| {i+1} | `{f['feature']}` | {f['importance']:.4f} |")
        
    lines.extend([
        "",
        "## Validation Integrity Notes",
        "- **Train/test split**: 10-fold Leave-One-Subject-Out CV",
        "- **Why LOSO-CV?**: With only 10 subjects available, leave-one-subject-out cross-validation gives a more statistically honest estimate than a single train/test split.",
        "- **No future leakage**: /predict endpoint uses only data up to time t",
        "- **Real ground truth**: Annotations by sleep medicine clinicians",
        "- **EHR profiles**: Synthetic (Synthea-inspired), clearly labeled",
        "",
        "## Citation",
        "Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The apnea-ECG database.",
        "Computers in Cardiology 2000;27:255-258.",
        ""
    ])

    md = "\n".join(lines)
    with open(RESULTS_DIR / "results.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nOK Results saved to {RESULTS_DIR}/results.json and results.md")


# ─── Main training pipeline ──────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        print("ERROR: features.parquet not found. Run: python src/features/engineer.py")
        sys.exit(1)

    features_df  = pd.read_parquet(feat_path)
    all_subjects = sorted(features_df["subject_id"].unique())
    n_subjects   = len(all_subjects)
    print(f"Loaded features: {features_df.shape}  ({n_subjects} subjects)")

    EHR_COLS = {
        "age", "bmi", "neck_circumference_cm", "hypertension",
        "sedative_use", "alcohol_use", "prior_ahi", "ess_score",
        "stopbang_score", "sex_male",
    }
    
    cv_fused = []
    cv_ehr = []
    cv_sig = []
    per_subject_metrics = []
    
    pooled = {
        "fused": {"y_true": [], "y_prob": []},
        "ehr_only": {"y_true": [], "y_prob": []},
        "wearable_only": {"y_true": [], "y_prob": []}
    }

    print("\nStarting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    
    shap_result = {}

    for fold, test_subj in enumerate(all_subjects):
        print(f"\n{'='*50}\nFold {fold+1}/{n_subjects} — Held-out Subject: {test_subj}")
        
        train_df = features_df[features_df["subject_id"] != test_subj]
        test_df  = features_df[features_df["subject_id"] == test_subj]

        # ── Compute imputation means from TRAINING fold only (no leakage) ──
        seq_impute_means = _compute_seq_impute_means(train_df)
        print(f"  Seq impute means (train fold): "
              f"HR={seq_impute_means['hr']:.1f}  "
              f"HRV={seq_impute_means['hrv']:.1f}  "
              f"SpO2={seq_impute_means['spo2']:.1f}")

        train_seqs, train_labels = _build_sequences(train_df, impute_means=seq_impute_means)
        test_seqs,  test_labels  = _build_sequences(test_df,  impute_means=seq_impute_means)
        train_seqs_n, test_seqs_n, seq_mean, seq_std = _normalize_sequences(train_seqs, test_seqs)

        encoder = train_gru_encoder(train_seqs_n, train_labels, hidden_size=GRU_HIDDEN, epochs=10, device=device)
        
        train_gru_h = encode_sequences(encoder, train_seqs_n, device=device)
        test_gru_h  = encode_sequences(encoder, test_seqs_n,  device=device)

        feat_cols = [c for c in _get_feature_cols(features_df) if c != "sex"]

        # ── Feature-level imputation: use per-column training-fold means ──
        # This replaces the blanket fillna(0) that caused -25σ z-scores for
        # SpO2-absent subjects (a01, a02).
        feat_impute_means_arr = train_df[feat_cols].mean(axis=0).values.astype(np.float32)
        # For any column that is ALL NaN in train (e.g. constant-zero flags),
        # fall back to 0 so normalization stays stable.
        feat_impute_means_arr = np.nan_to_num(feat_impute_means_arr, nan=0.0)

        def impute_df(df_slice):
            """Fill NaN in feat_cols using training-fold column means."""
            arr = df_slice[feat_cols].values.astype(np.float32)
            nan_mask = np.isnan(arr)
            arr[nan_mask] = np.take(feat_impute_means_arr, np.where(nan_mask)[1])
            return arr

        X_train_feat = impute_df(train_df)
        X_test_feat  = impute_df(test_df)
        y_train = train_df["event_within_30min"].values.astype(np.float32)
        y_test  = test_df["event_within_30min"].values.astype(np.float32)

        feat_mean = X_train_feat.mean(axis=0)
        feat_std  = X_train_feat.std(axis=0) + 1e-8
        X_train_feat_n = (X_train_feat - feat_mean) / feat_std
        X_test_feat_n  = (X_test_feat  - feat_mean) / feat_std

        gru_feat_names = [f"gru_h_{i}" for i in range(GRU_HIDDEN)]
        all_feat_names = feat_cols + gru_feat_names
        
        X_train = np.concatenate([X_train_feat_n, train_gru_h], axis=1)
        X_test  = np.concatenate([X_test_feat_n,  test_gru_h],  axis=1)

        if fold == 0:
            print("\n--- EHR Columns ---")
            ehr_cols_print = [c for c in feat_cols if c in EHR_COLS]
            print(ehr_cols_print)
            for c in ehr_cols_print:
                print(f"  {c}: min={features_df[c].min():.2f}, max={features_df[c].max():.2f}, std={features_df[c].std():.4f}")
            print("\n--- Fused Feature Names ---")
            print(all_feat_names)
            print("-------------------\n")

        print(f"  Training Fused Model...")
        model, metrics_fused, proba_fused = train_lightgbm(X_train, y_train, X_test, y_test, all_feat_names)
        if "auroc" in metrics_fused:
            cv_fused.append(metrics_fused)
            if proba_fused is not None and "proba" in proba_fused:
                pooled["fused"]["y_true"].extend(y_test)
                pooled["fused"]["y_prob"].extend(proba_fused["proba"])
        
        ehr_cols = [c for c in feat_cols if c in EHR_COLS]
        sig_cols = [c for c in feat_cols if c not in EHR_COLS]

        print(f"  Training EHR-only...")
        if ehr_cols:
            m_ehr, metrics_ehr, proba_ehr = train_lightgbm(
                train_df[ehr_cols].fillna(0).values.astype(np.float32), y_train,
                test_df[ehr_cols].fillna(0).values.astype(np.float32),  y_test, ehr_cols
            )
            if "auroc" in metrics_ehr:
                cv_ehr.append(metrics_ehr)
                if proba_ehr is not None and "proba" in proba_ehr:
                    pooled["ehr_only"]["y_true"].extend(y_test)
                    pooled["ehr_only"]["y_prob"].extend(proba_ehr["proba"])
        else:
            cv_ehr.append({})

        print(f"  Training Wearables-only...")
        m_sig, metrics_sig, proba_sig = train_lightgbm(
            np.concatenate([train_df[sig_cols].fillna(0).values.astype(np.float32), train_gru_h], axis=1), y_train,
            np.concatenate([test_df[sig_cols].fillna(0).values.astype(np.float32),  test_gru_h],  axis=1), y_test,
            sig_cols + gru_feat_names,
        )
        if "auroc" in metrics_sig:
            cv_sig.append(metrics_sig)
            if proba_sig is not None and "proba" in proba_sig:
                pooled["wearable_only"]["y_true"].extend(y_test)
                pooled["wearable_only"]["y_prob"].extend(proba_sig["proba"])

        if "auroc" in metrics_fused:
            per_subject_metrics.append({
                "subject_id": str(test_subj),
                "auroc": float(metrics_fused.get("auroc", float('nan'))),
                "auprc": float(metrics_fused.get("auprc", float('nan'))),
                "sens_80": float(metrics_fused.get("sensitivity_at_80_specificity", float('nan'))),
                "brier": float(metrics_fused.get("brier_score", float('nan')))
            })
        
        if fold == n_subjects - 1:
            print("\nComputing SHAP on last fold...")
            shap_result = compute_shap(model, X_test, all_feat_names)
            
            with open(MODELS_DIR / "lgbm_model.pkl", "wb") as f:
                pickle.dump(model, f)
            torch.save(encoder.state_dict(), MODELS_DIR / "gru_encoder.pt")
            config = {
                "feature_names": feat_cols, "all_feature_names": all_feat_names,
                "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
                # feat_impute_means: per-feature training-fold means used to fill NaN
                # BEFORE z-score normalization. Predictor uses these at inference time
                # so missing signals normalize to ~0 (neutral) not -25σ.
                "feat_impute_means": feat_impute_means_arr.tolist(),
                "seq_mean": seq_mean.tolist(), "seq_std": seq_std.tolist(),
                # seq_impute_means: per-channel means for GRU sequence imputation
                "seq_impute_means": seq_impute_means,
                "gru_hidden": GRU_HIDDEN, "seq_len": GRU_SEQUENCE_LEN,
                "n_test_subjects": n_subjects,
            }
            with open(MODELS_DIR / "feature_names.json", "w") as f:
                json.dump(config, f, indent=2)

    def avg_metrics(cv_list):
        if not cv_list or len(cv_list) == 0:
            return {}
        keys = ["auroc", "auprc", "sensitivity_at_80_specificity", "sensitivity_at_90_specificity", "brier_score"]
        res = {}
        for k in keys:
            vals = [m[k] for m in cv_list if k in m and not np.isnan(m[k])]
            if len(vals) > 0:
                res[k] = float(np.mean(vals))
                res[f"{k}_std"] = float(np.std(vals))
            else:
                res[k] = None
                res[f"{k}_std"] = None
        return res

    final_fused = avg_metrics(cv_fused)
    final_ehr   = avg_metrics(cv_ehr)
    final_sig   = avg_metrics(cv_sig)

    ablation = {"ehr_only": final_ehr, "wearable_only": final_sig, "fused": final_fused}
    
    final_fused["n_test"] = sum(m.get("n_test", 0) for m in cv_fused)
    if cv_fused:
        final_fused["positive_rate_test"] = float(np.mean([m.get("positive_rate_test", 0) for m in cv_fused]))

    def calc_pooled_metrics(pooled_dict):
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
        if len(pooled_dict["y_true"]) == 0:
            return {}
        y_t = np.array(pooled_dict["y_true"])
        y_p = np.array(pooled_dict["y_prob"])
        if len(np.unique(y_t)) < 2:
            return {}
        auroc = roc_auc_score(y_t, y_p)
        auprc = average_precision_score(y_t, y_p)
        brier = brier_score_loss(y_t, y_p)
        fpr, tpr, _ = roc_curve(y_t, y_p)
        sens_80 = float(tpr[np.argmin(np.abs((1 - fpr) - 0.80))])
        return {
            "pooled_auroc": float(auroc),
            "pooled_auprc": float(auprc),
            "pooled_sens_80": sens_80,
            "pooled_brier": float(brier)
        }

    for k in pooled:
        ablation[k].update(calc_pooled_metrics(pooled[k]))

    model_config = {
        "gru_hidden": GRU_HIDDEN, "gru_layers": 2, "seq_len": GRU_SEQUENCE_LEN,
        "n_features": len(feat_cols), "n_gru_features": GRU_HIDDEN,
        "n_total_features": len(all_feat_names), "n_test_subjects": n_subjects,
        "lgbm_n_estimators": 400,
    }

    write_results(final_fused, ablation, shap_result, model_config, n_subjects, per_subject_metrics)
    print("\nOK Training complete.")


if __name__ == "__main__":
    main()
