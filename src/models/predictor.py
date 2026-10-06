"""
src/models/predictor.py
-----------------------
Inference wrapper that loads trained model artifacts and provides:
  - predict_at_minute(subject_df, minute, ehr_row) → (risk_score, shap_reasons)
  - predict_simulate(hr, hrv, spo2, ehr_row) → (risk_score, shap_reasons)

Used by the FastAPI endpoints.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import torch

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import SignalGRUEncoder, GRU_HIDDEN, GRU_SEQUENCE_LEN
from src.features.engineer import engineer_features_for_subject, WINDOWS

DEVICE = torch.device("cpu")  # CPU for inference


class SomnusTwinPredictor:
    """Loaded inference engine — singleton-friendly."""

    _instance = None

    def __init__(self):
        self._loaded = False
        self.model = None
        self.encoder = None
        self.config = None
        self.explainer = None

    @classmethod
    def get(cls) -> "SomnusTwinPredictor":
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._loaded:
            cls._instance.load()
        return cls._instance

    def load(self) -> None:
        """Load model artifacts from disk."""
        try:
            with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open(MODELS_DIR / "feature_names.json") as f:
                self.config = json.load(f)

            self.encoder = SignalGRUEncoder(hidden_size=GRU_HIDDEN).to(DEVICE)
            self.encoder.load_state_dict(
                torch.load(MODELS_DIR / "gru_encoder.pt", map_location=DEVICE)
            )
            self.encoder.eval()

            # SHAP explainer (TreeExplainer is fast for LightGBM)
            self.explainer = shap.TreeExplainer(self.model)

            self._loaded = True
            print("OK SomnusTwinPredictor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load trained model: {e}")
            print("  Run training pipeline first (src/models/train.py)")
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def _normalize_features(self, feat_vec: np.ndarray) -> np.ndarray:
        mean = np.array(self.config["feat_mean"])
        std = np.array(self.config["feat_std"])
        normed = (feat_vec - mean) / (std + 1e-8)
        return np.nan_to_num(normed, nan=0.0)

    def _normalize_seq(self, seq: np.ndarray) -> np.ndarray:
        mean = np.array(self.config["seq_mean"])
        std = np.array(self.config["seq_std"])
        normed = (seq - mean) / (std + 1e-8)
        return np.nan_to_num(normed, nan=0.0)

    def _encode_gru(self, seq: np.ndarray) -> np.ndarray:
        """seq: (seq_len, 3)"""
        seq_norm = self._normalize_seq(seq)
        t = torch.tensor(seq_norm[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            h = self.encoder(t).cpu().numpy()[0]
        return h

    def _build_feature_vector(
        self,
        signal_df: pd.DataFrame,
        minute: int,
        ehr_row: Optional[pd.Series] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (feature_vector, sequence) for a given minute."""
        # Slice signal up to this minute (no future leakage)
        df_up_to_t = signal_df[signal_df["minute"] <= minute].reset_index(drop=True)

        if len(df_up_to_t) == 0:
            n_feats = len(self.config["feature_names"])
            return np.zeros(n_feats), np.zeros((GRU_SEQUENCE_LEN, 3))

        # Re-engineer features for this subject up to minute t
        feat_df = engineer_features_for_subject(df_up_to_t, ehr_row)
        t_local = min(minute, len(feat_df) - 1)
        feat_row = feat_df.iloc[t_local]

        feat_names = self.config["feature_names"]

        # Per-feature imputation means from the training fold (saved during training).
        # Fall back to feat_mean (post-normalization anchor) if key absent in older configs.
        feat_impute = self.config.get("feat_impute_means", self.config["feat_mean"])

        feat_vec = np.array([
            feat_row.get(n, feat_impute[i])
            if not pd.isna(feat_row.get(n, np.nan))
            else feat_impute[i]       # use training-fold mean, NOT 0
            for i, n in enumerate(feat_names)
        ], dtype=np.float32)

        # Build GRU sequence with per-channel training-fold imputation
        sim = self.config.get("seq_impute_means", {"hr": 70.0, "hrv": 25.0, "spo2": 95.0})
        hr_imp, hrv_imp, spo2_imp = sim["hr"], sim["hrv"], sim["spo2"]

        hr   = df_up_to_t["hr_bpm"].fillna(hr_imp).values
        hrv  = df_up_to_t["hrv_rmssd"].fillna(hrv_imp).values
        spo2 = df_up_to_t["spo2"].fillna(spo2_imp).values
        seq_len = GRU_SEQUENCE_LEN
        start = max(0, len(hr) - seq_len)
        hr_seg   = hr[start:];   hrv_seg = hrv[start:];   spo2_seg = spo2[start:]
        pad = seq_len - len(hr_seg)
        hr_seg   = np.concatenate([np.full(pad, hr_imp),   hr_seg])
        hrv_seg  = np.concatenate([np.full(pad, hrv_imp),  hrv_seg])
        spo2_seg = np.concatenate([np.full(pad, spo2_imp), spo2_seg])
        seq = np.stack([hr_seg, hrv_seg, spo2_seg], axis=1).astype(np.float32)

        return feat_vec, seq

    def _shap_reasons(
        self, feat_vec_normalized: np.ndarray, top_n: int = 5
    ) -> list[dict]:
        """Return top SHAP-driven reason codes as plain-language strings."""
        try:
            shap_vals = self.explainer.shap_values(feat_vec_normalized.reshape(1, -1))
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # binary positive class
            shap_vals = shap_vals[0]
            all_feat_names = self.config["all_feature_names"]

            # Sort by absolute contribution
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

            reasons = []
            for idx in top_idx:
                name = all_feat_names[idx]
                val = shap_vals[idx]
                reasons.append({
                    "feature": name,
                    "shap_value": float(val),
                    "direction": "^ risk" if val > 0 else "v risk",
                    "label": _feature_to_label(name, val),
                })
            return reasons
        except Exception:
            return []

    def predict_at_minute(
        self,
        signal_df: pd.DataFrame,
        minute: int,
        ehr_row: Optional[pd.Series] = None,
    ) -> dict:
        """
        Predict risk at a specific minute using only data up to that minute.
        Returns: {"risk_score": float, "reasons": list, "minute": int}
        """
        if not self.is_ready:
            return self._fallback_predict(signal_df, minute)

        feat_vec, seq = self._build_feature_vector(signal_df, minute, ehr_row)
        feat_vec_norm = self._normalize_features(feat_vec)
        gru_h = self._encode_gru(seq)

        X = np.concatenate([feat_vec_norm, gru_h]).reshape(1, -1)
        risk = float(self.model.predict_proba(X)[0, 1])

        all_feat_vec = np.concatenate([feat_vec_norm, gru_h])
        reasons = self._shap_reasons(all_feat_vec)

        return {"risk_score": risk, "reasons": reasons, "minute": minute}

    def predict_simulate(
        self,
        hr: float,
        hrv: float,
        spo2: float,
        ehr_data: Optional[dict] = None,
    ) -> dict:
        """
        Live "what-if" prediction from manual inputs.
        Builds a minimal synthetic signal window and runs the real model.
        """
        if not self.is_ready:
            return {"risk_score": float(np.clip((100 - spo2) * 0.05 + (hr - 65) * 0.01, 0, 1)),
                    "reasons": [], "minute": -1}

        # Build a minimal feature vector from manual inputs
        ehr_series = pd.Series(ehr_data) if ehr_data else None
        
        # Create a synthetic 30-minute signal with the given steady-state values
        n = GRU_SEQUENCE_LEN
        hr_arr = np.full(n, hr if not np.isnan(hr) else 70.0)
        hrv_arr = np.full(n, hrv if not np.isnan(hrv) else 25.0)
        spo2_arr = np.full(n, spo2 if not np.isnan(spo2) else 97.0)
        seq = np.stack([hr_arr, hrv_arr, spo2_arr], axis=1).astype(np.float32)

        # Build minimal feature row from manual inputs
        feat_row = _manual_inputs_to_features(hr, hrv, spo2, ehr_data or {})
        feat_names = self.config["feature_names"]
        feat_vec = np.array([feat_row.get(n, 0.0) for n in feat_names], dtype=np.float32)

        feat_vec_norm = self._normalize_features(feat_vec)
        gru_h = self._encode_gru(seq)

        X = np.concatenate([feat_vec_norm, gru_h]).reshape(1, -1)
        risk = float(self.model.predict_proba(X)[0, 1])

        all_feat_vec = np.concatenate([feat_vec_norm, gru_h])
        reasons = self._shap_reasons(all_feat_vec)

        return {"risk_score": risk, "reasons": reasons, "minute": -1}

    def _fallback_predict(self, signal_df: pd.DataFrame, minute: int) -> dict:
        """Heuristic fallback when model artifacts not loaded."""
        row = signal_df[signal_df["minute"] <= minute]
        if row.empty:
            return {"risk_score": 0.1, "reasons": [], "minute": minute}
        last = row.iloc[-1]
        spo2 = last.get("spo2", 97.0) or 97.0
        hr = last.get("hr_bpm", 70.0) or 70.0
        risk = float(np.clip((100 - spo2) * 0.04 + max(0, hr - 80) * 0.005, 0.05, 0.95))
        return {"risk_score": risk, "reasons": [], "minute": minute}


def _feature_to_label(feature: str, shap_val: float) -> str:
    """Convert feature name + SHAP direction to a plain-language reason code."""
    direction = "elevated" if shap_val > 0 else "reduced"
    label_map = {
        "spo2_mean": f"SpO₂ {direction} in recent window",
        "spo2_min": f"SpO₂ minimum {direction} — possible desaturation",
        "spo2_trend": f"SpO₂ trend {'declining' if shap_val > 0 else 'stable'}",
        "spo2_dips": f"Desaturation dip count {'high' if shap_val > 0 else 'low'}",
        "hr_mean": f"Heart rate {direction}",
        "hr_trend": f"Heart rate {'rising' if shap_val > 0 else 'falling'}",
        "hrv_mean": f"HRV {direction} — {'autonomic stress' if shap_val > 0 else 'good autonomic tone'}",
        "hrv_trend": f"HRV trend {'worsening' if shap_val > 0 else 'improving'}",
        "minutes_since_last_apnea": f"Recent apnea event {'close' if shap_val > 0 else 'distant'}",
        "apnea_count_so_far": f"Cumulative apnea burden {'high' if shap_val > 0 else 'low'}",
        "bmi": f"BMI {'high — OSA risk factor' if shap_val > 0 else 'within normal range'}",
        "prior_ahi": f"Prior AHI {'high — established OSA' if shap_val > 0 else 'low'}",
        "stopbang_score": f"STOP-BANG score {'high — clinical risk' if shap_val > 0 else 'low'}",
        "hypertension": f"Hypertension {'present — comorbid risk' if shap_val > 0 else 'absent'}",
        "sedative_use": f"Sedative use {'present — suppressed arousals' if shap_val > 0 else 'absent'}",
    }
    for key, label in label_map.items():
        if key in feature:
            return label
    if feature.startswith("gru_h_"):
        return f"Neural signal pattern ({direction} activation)"
    return f"{feature.replace('_', ' ').title()} {direction}"


def _manual_inputs_to_features(hr: float, hrv: float, spo2: float, ehr: dict) -> dict:
    """Build a minimal feature dict from manual sliders for simulate endpoint."""
    feats = {
        # Current values
        "hr_current": hr, "hrv_current": hrv, "spo2_current": spo2,
        # Use same value for all rolling windows (no history)
        **{f"hr_mean_{w}m": hr for w in WINDOWS},
        **{f"hr_std_{w}m": 0.0 for w in WINDOWS},
        **{f"hr_trend_{w}m": 0.0 for w in WINDOWS},
        **{f"hr_missing_{w}m": 0.0 for w in WINDOWS},
        **{f"hrv_mean_{w}m": hrv for w in WINDOWS},
        **{f"hrv_trend_{w}m": 0.0 for w in WINDOWS},
        **{f"hrv_missing_{w}m": 0.0 for w in WINDOWS},
        **{f"spo2_mean_{w}m": spo2 for w in WINDOWS},
        **{f"spo2_trend_{w}m": 0.0 for w in WINDOWS},
        **{f"spo2_dips_{w}m": 0.0 for w in WINDOWS},
        **{f"spo2_missing_{w}m": 0.0 for w in WINDOWS},
        **{f"spo2_min_{w}m": spo2 for w in WINDOWS},
        "minutes_since_last_apnea": -1,
        "apnea_count_so_far": 0,
        # EHR
        "age": ehr.get("age", 50),
        "bmi": ehr.get("bmi", 28),
        "neck_circumference_cm": ehr.get("neck_circumference_cm", 40),
        "hypertension": ehr.get("hypertension", 0),
        "sedative_use": ehr.get("sedative_use", 0),
        "alcohol_use": ehr.get("alcohol_use", 0),
        "prior_ahi": ehr.get("prior_ahi", 10),
        "ess_score": ehr.get("ess_score", 8),
        "stopbang_score": ehr.get("stopbang_score", 3),
        "sex_male": 1 if ehr.get("sex", "M") == "M" else 0,
    }
    return feats
