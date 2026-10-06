"""
src/api/main.py
---------------
SomnusTwin FastAPI backend.

Endpoints:
  GET  /                              → health check
  GET  /subjects                      → list demo subjects with metadata
  GET  /subjects/{id}/night           → full aligned night signal + annotations
  GET  /subjects/{id}/twin-state      → full twin-state trajectory
  GET  /subjects/{id}/predict?t=MIN   → prediction AT minute t (no future leakage)
  POST /simulate                      → live "what-if" prediction

CORS is open for local development (frontend on :5173 → backend on :8000).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.predictor import SomnusTwinPredictor
from src.twin.state import PatientDigitalTwin

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data"


# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SomnusTwin API",
    description=(
        "OSA desaturation prediction digital twin — "
        "trained on real PhysioNet clinician-scored apnea annotations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup: load predictor + cache subject manifests ───────────────────────

subjects_df: Optional[pd.DataFrame] = None
ehr_df: Optional[pd.DataFrame] = None
predictor: Optional[SomnusTwinPredictor] = None


@app.on_event("startup")
async def startup_event():
    global subjects_df, ehr_df, predictor

    subjects_path = PROCESSED_DIR / "subjects.parquet"
    ehr_path = PROCESSED_DIR / "ehr_profiles.parquet"

    if subjects_path.exists():
        subjects_df = pd.read_parquet(subjects_path)
        print(f"OK Loaded {len(subjects_df)} subjects")
    else:
        print("WARN subjects.parquet not found. Run fetch_physionet.py first.")
        subjects_df = pd.DataFrame()

    if ehr_path.exists():
        ehr_df = pd.read_parquet(ehr_path)
        print(f"OK Loaded EHR profiles for {len(ehr_df)} subjects")
    else:
        ehr_df = pd.DataFrame()

    predictor = SomnusTwinPredictor.get()


# ─── Helper utilities ────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if np.isnan(v) or np.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _load_subject_signal(subject_id: str) -> pd.DataFrame:
    sig_path = PROCESSED_DIR / f"signals_{subject_id}.parquet"
    if not sig_path.exists():
        raise HTTPException(status_code=404, detail=f"Signal data for {subject_id} not found")
    return pd.read_parquet(sig_path)


def _get_ehr_row(subject_id: str) -> Optional[pd.Series]:
    if ehr_df is None or ehr_df.empty:
        return None
    mask = ehr_df["subject_id"] == subject_id
    if not mask.any():
        return None
    return ehr_df[mask].iloc[0]


def _get_subject_meta(subject_id: str) -> dict:
    if subjects_df is None or subjects_df.empty:
        return {}
    mask = subjects_df["subject_id"] == subject_id
    if not mask.any():
        return {}
    row = subjects_df[mask].iloc[0]
    return {
        "subject_id": row["subject_id"],
        "n_minutes": int(row["n_minutes"]),
        "total_apnea_minutes": int(row["total_apnea_minutes"]),
        "ahi_approx": float(row["ahi_approx"]),
        "osa_severity": str(row["osa_severity"]),
        "has_spo2": bool(row["has_spo2"]),
    }


# ─── Pydantic models ─────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    hr: float = 75.0
    hrv: float = 25.0
    spo2: float = 97.0
    age: Optional[float] = 50
    bmi: Optional[float] = 28
    neck_circumference_cm: Optional[float] = 40
    hypertension: Optional[int] = 0
    sedative_use: Optional[int] = 0
    alcohol_use: Optional[int] = 0
    prior_ahi: Optional[float] = 10
    ess_score: Optional[int] = 8
    stopbang_score: Optional[int] = 3
    sex: Optional[str] = "M"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    import os
    model_mtime = None
    model_path = PROCESSED_DIR.parent / "models" / "lgbm_model.pkl"
    if model_path.exists():
        model_mtime = os.path.getmtime(model_path)
    return {
        "status": "ok",
        "service": "SomnusTwin API",
        "model_ready": predictor.is_ready if predictor else False,
        "n_subjects": len(subjects_df) if subjects_df is not None else 0,
        "data_source": "PhysioNet Apnea-ECG Database (real clinician-scored annotations)",
        "model_mtime": model_mtime,
    }


@app.get("/subjects")
def list_subjects():
    """Return all available demo subjects with EHR and severity metadata."""
    if subjects_df is None or subjects_df.empty:
        return {"subjects": [], "error": "No data available. Run fetch_physionet.py first."}

    result = []
    for _, row in subjects_df.iterrows():
        sid = str(row["subject_id"])
        ehr_row = _get_ehr_row(sid)

        subject = {
            "subject_id": sid,
            "n_minutes": int(row["n_minutes"]),
            "total_apnea_minutes": int(row["total_apnea_minutes"]),
            "ahi_approx": float(row["ahi_approx"]),
            "osa_severity": str(row["osa_severity"]),
            "has_spo2": bool(row["has_spo2"]),
            # EHR (synthetic)
            "age": int(ehr_row["age"]) if ehr_row is not None else None,
            "bmi": float(ehr_row["bmi"]) if ehr_row is not None else None,
            "sex": str(ehr_row["sex"]) if ehr_row is not None else None,
            "hypertension": bool(ehr_row["hypertension"]) if ehr_row is not None else None,
            "prior_ahi": float(ehr_row["prior_ahi"]) if ehr_row is not None else None,
            "ess_score": int(ehr_row["ess_score"]) if ehr_row is not None else None,
            "stopbang_score": int(ehr_row["stopbang_score"]) if ehr_row is not None else None,
            "cpap_prescribed": bool(ehr_row["cpap_prescribed"]) if ehr_row is not None else None,
        }
        result.append(subject)

    # Sort by severity (severe first)
    severity_order = {"severe": 0, "moderate": 1, "mild": 2, "normal": 3}
    result.sort(key=lambda x: severity_order.get(x["osa_severity"], 4))

    return {"subjects": result, "n_subjects": len(result)}


@app.get("/subjects/{subject_id}/night")
def get_night_signal(subject_id: str):
    """
    Return full aligned night signal + real annotated events for a subject.
    This is the raw data that powers the Night Playback Mode.
    """
    sig_df = _load_subject_signal(subject_id)
    meta = _get_subject_meta(subject_id)

    # Build per-minute signal data
    signal_data = []
    for _, row in sig_df.iterrows():
        signal_data.append({
            "minute": int(row["minute"]),
            "hr": _safe_float(row.get("hr_bpm")),
            "hrv": _safe_float(row.get("hrv_rmssd")),
            "spo2": _safe_float(row.get("spo2")),
            "apnea_label": int(row.get("apnea_label", 0)),
        })

    # Extract annotated event intervals for timeline markers
    apnea_events = []
    in_event = False
    event_start = None
    for row in signal_data:
        if row["apnea_label"] == 1 and not in_event:
            in_event = True
            event_start = row["minute"]
        elif row["apnea_label"] == 0 and in_event:
            in_event = False
            apnea_events.append({
                "start_minute": event_start,
                "end_minute": row["minute"] - 1,
                "duration_minutes": row["minute"] - event_start,
                "type": "annotated_apnea",
                "source": "PhysioNet_clinician_scored",
            })
    if in_event:
        apnea_events.append({
            "start_minute": event_start,
            "end_minute": signal_data[-1]["minute"],
            "duration_minutes": signal_data[-1]["minute"] - event_start + 1,
            "type": "annotated_apnea",
            "source": "PhysioNet_clinician_scored",
        })

    return {
        "subject_id": subject_id,
        "meta": meta,
        "signal": signal_data,
        "annotated_events": apnea_events,
        "total_apnea_minutes": meta.get("total_apnea_minutes", 0),
        "data_source": "PhysioNet Apnea-ECG Database (real, clinician-scored)",
    }


@app.get("/subjects/{subject_id}/twin-state")
def get_twin_state(
    subject_id: str,
    start_minute: int = Query(0, ge=0),
    end_minute: Optional[int] = Query(None, ge=0),
):
    """
    Return the persisted twin-state trajectory for a subject.
    Falls back to computing from signal + predictor if not pre-populated.
    """
    twin = PatientDigitalTwin(subject_id=subject_id)
    trajectory = twin.get_trajectory(start_minute, end_minute)

    if not trajectory:
        # Compute states on-the-fly from raw signal (heuristic or model).
        sig_df  = _load_subject_signal(subject_id)
        ehr_row = _get_ehr_row(subject_id)

        result = []
        for idx in range(start_minute, len(sig_df)):
            row_data = sig_df.iloc[idx]
            minute   = int(row_data.get("minute", idx))
            if end_minute is not None and minute > end_minute:
                break
            spo2  = _safe_float(row_data.get("spo2"))      or 97.0
            hr    = _safe_float(row_data.get("hr_bpm"))    or 70.0
            hrv   = _safe_float(row_data.get("hrv_rmssd")) or 30.0
            apnea = int(row_data.get("apnea_label", 0))

            if predictor and predictor.is_ready:
                pred = predictor.predict_at_minute(sig_df, minute, ehr_row)
                risk = pred["risk_score"]
            else:
                risk = float(np.clip(
                    max(0, (100 - spo2) * 0.045) +
                    max(0, (hr - 70)   * 0.006)  +
                    max(0, (30 - hrv)  * 0.004),
                    0.03, 0.95
                ))

            result.append({
                "minute":          minute,
                "risk_score":      round(risk, 4),
                "hr_current":      round(hr,   1),
                "hrv_current":     round(hrv,  1),
                "spo2_current":    round(spo2, 1),
                "apnea_now":       apnea,
                "gru_hidden_norm": None,
                "created_at":      None,
                "state_vector":    {},
            })

        # Persist to SQLite so next call is instant
        twin_writer = PatientDigitalTwin(subject_id=subject_id)
        for rec in result:
            twin_writer.update(
                minute=rec["minute"],
                feature_row={
                    "hr_current":   rec["hr_current"],
                    "hrv_current":  rec["hrv_current"],
                    "spo2_current": rec["spo2_current"],
                    "apnea_now":    rec["apnea_now"],
                },
                risk_score=rec["risk_score"],
            )

        return {
            "subject_id": subject_id,
            "trajectory":  result,
            "n_states":    len(result),
            "source":      "model" if (predictor and predictor.is_ready) else "heuristic",
        }

    return {
        "subject_id": subject_id,
        "trajectory":  trajectory,
        "n_states":    len(trajectory),
    }


@app.get("/subjects/{subject_id}/predict")
def predict_at_time(
    subject_id: str,
    t: int = Query(..., ge=0, description="Minute of the night to predict at"),
):
    """
    Return the model's risk prediction AT minute t,
    using ONLY signal data up to and including minute t.
    This is what powers Night Playback Mode — no future leakage.
    """
    sig_df = _load_subject_signal(subject_id)
    ehr_row = _get_ehr_row(subject_id)

    if t > sig_df["minute"].max():
        raise HTTPException(status_code=400, detail=f"Minute {t} exceeds night length")

    if not predictor or not predictor.is_ready:
        # Graceful fallback
        row = sig_df[sig_df["minute"] <= t]
        if row.empty:
            return {"risk_score": 0.1, "reasons": [], "minute": t, "fallback": True}
        last = row.iloc[-1]
        spo2 = _safe_float(last.get("spo2")) or 97.0
        hr = _safe_float(last.get("hr_bpm")) or 70.0
        risk = float(np.clip((100 - spo2) * 0.04 + max(0, hr - 80) * 0.005, 0.05, 0.95))
        return {"risk_score": risk, "reasons": [], "minute": t, "fallback": True}

    result = predictor.predict_at_minute(sig_df, t, ehr_row)

    # Also return the actual ground-truth status at this minute (for the "predicted N min early" badge)
    row_at_t = sig_df[sig_df["minute"] == t]
    apnea_now = int(row_at_t.iloc[0]["apnea_label"]) if not row_at_t.empty else 0

    # Check if an annotated event starts within next 30 min
    future = sig_df[(sig_df["minute"] > t) & (sig_df["minute"] <= t + 30)]
    real_event_imminent = int(future["apnea_label"].sum() > 0)

    result["apnea_now"] = apnea_now
    result["real_event_within_30min"] = real_event_imminent
    result["data_source"] = "PhysioNet Apnea-ECG annotations"

    return result


@app.post("/simulate")
def simulate(request: SimulateRequest):
    """
    Live 'what-if' prediction from manual HR/HRV/SpO2 inputs.
    Calls the real trained model — proves it's not hardcoded.
    """
    ehr_data = {
        "age": request.age,
        "bmi": request.bmi,
        "neck_circumference_cm": request.neck_circumference_cm,
        "hypertension": request.hypertension,
        "sedative_use": request.sedative_use,
        "alcohol_use": request.alcohol_use,
        "prior_ahi": request.prior_ahi,
        "ess_score": request.ess_score,
        "stopbang_score": request.stopbang_score,
        "sex": request.sex,
    }

    if not predictor or not predictor.is_ready:
        # Heuristic fallback
        spo2_risk = max(0, (100 - request.spo2) * 0.05)
        hr_risk = max(0, (request.hr - 65) * 0.008)
        risk = float(np.clip(spo2_risk + hr_risk, 0.02, 0.98))
        return {"risk_score": risk, "reasons": [], "fallback": True}

    result = predictor.predict_simulate(
        hr=request.hr,
        hrv=request.hrv,
        spo2=request.spo2,
        ehr_data=ehr_data,
    )
    result["inputs"] = {"hr": request.hr, "hrv": request.hrv, "spo2": request.spo2}
    return result


@app.get("/validation")
async def get_validation_results():
    """Return the validation report for the Results page."""
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        return {
            "error": "results.json not found. Run training pipeline first.",
            "instructions": "Run: python src/models/train.py",
        }
    with open(results_path) as f:
        return json.load(f)


@app.get("/dataset-info")
async def get_dataset_info():
    """Return dataset metadata for the About panel."""
    summary_path = PROCESSED_DIR / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            info = json.load(f)
    else:
        info = {
            "data_sources": {
                "signals": "PhysioNet Apnea-ECG Database",
                "ehr_profiles": "Synthetic (Synthea-inspired)",
            }
        }

    # Augment with live stats
    if subjects_df is not None and not subjects_df.empty:
        info["live_stats"] = {
            "n_subjects": len(subjects_df),
            "total_minutes": int(subjects_df["n_minutes"].sum()),
            "total_apnea_minutes": int(subjects_df["total_apnea_minutes"].sum()),
            "severity_distribution": subjects_df["osa_severity"].value_counts().to_dict(),
        }

    return info
