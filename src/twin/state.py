"""
src/twin/state.py
-----------------
PatientDigitalTwin — persists the versioned, timestamped twin-state
vector per subject into SQLite.  The state vector fuses:
  • Static EHR profile
  • Rolling dynamic signal features (per minute)
  • GRU hidden-state summary of the trailing signal window

The full nightly trajectory (one state snapshot per minute) is
queryable and replayable — this IS the "Digital Twin".
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DB_PATH = Path(__file__).parent.parent.parent / "data" / "twin_states.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class TwinStateRecord(Base):
    """SQLAlchemy ORM model for a single timestep of a patient's twin state."""
    __tablename__ = "twin_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(20), nullable=False, index=True)
    minute = Column(Integer, nullable=False)
    risk_score = Column(Float, nullable=True)
    hr_current = Column(Float, nullable=True)
    hrv_current = Column(Float, nullable=True)
    spo2_current = Column(Float, nullable=True)
    apnea_now = Column(Integer, nullable=True)
    gru_hidden_norm = Column(Float, nullable=True)   # ‖h_t‖ as a scalar summary
    state_vector_json = Column(Text, nullable=True)  # full JSON-serialized state
    created_at = Column(String(30), nullable=False)
    version = Column(Integer, default=1)


Base.metadata.create_all(engine)


class PatientDigitalTwin:
    """
    Persisted digital twin for a single patient.

    Usage:
        twin = PatientDigitalTwin(subject_id="a01")
        twin.update(minute=5, feature_row=feat_dict, risk_score=0.32)
        trajectory = twin.get_trajectory()
    """

    def __init__(self, subject_id: str):
        self.subject_id = subject_id

    def update(
        self,
        minute: int,
        feature_row: dict[str, Any],
        risk_score: float,
        gru_hidden: Optional[np.ndarray] = None,
    ) -> None:
        """Upsert one minute's twin state into SQLite."""
        gru_norm = float(np.linalg.norm(gru_hidden)) if gru_hidden is not None else None

        # Trim state vector to key fields for JSON storage
        state_vec = {
            k: (float(v) if isinstance(v, (float, np.floating)) else
                int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in feature_row.items()
            if not isinstance(v, (list, np.ndarray))
        }
        state_vec["risk_score"] = float(risk_score)

        with SessionLocal() as session:
            # Check if record exists (upsert pattern)
            existing = session.query(TwinStateRecord).filter_by(
                subject_id=self.subject_id, minute=minute
            ).first()

            if existing:
                existing.risk_score = risk_score
                existing.hr_current = feature_row.get("hr_current")
                existing.hrv_current = feature_row.get("hrv_current")
                existing.spo2_current = feature_row.get("spo2_current")
                existing.apnea_now = feature_row.get("apnea_now")
                existing.gru_hidden_norm = gru_norm
                existing.state_vector_json = json.dumps(state_vec)
                existing.created_at = datetime.now(timezone.utc).isoformat()
            else:
                record = TwinStateRecord(
                    subject_id=self.subject_id,
                    minute=minute,
                    risk_score=risk_score,
                    hr_current=feature_row.get("hr_current"),
                    hrv_current=feature_row.get("hrv_current"),
                    spo2_current=feature_row.get("spo2_current"),
                    apnea_now=feature_row.get("apnea_now"),
                    gru_hidden_norm=gru_norm,
                    state_vector_json=json.dumps(state_vec),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                session.add(record)
            session.commit()

    def get_trajectory(self, start_minute: int = 0, end_minute: Optional[int] = None) -> list[dict]:
        """Return the full (or partial) nightly trajectory as a list of dicts."""
        with SessionLocal() as session:
            q = session.query(TwinStateRecord).filter_by(subject_id=self.subject_id)
            q = q.filter(TwinStateRecord.minute >= start_minute)
            if end_minute is not None:
                q = q.filter(TwinStateRecord.minute <= end_minute)
            records = q.order_by(TwinStateRecord.minute).all()

        result = []
        for r in records:
            state_vec = json.loads(r.state_vector_json) if r.state_vector_json else {}
            result.append({
                "minute": r.minute,
                "risk_score": r.risk_score,
                "hr_current": r.hr_current,
                "hrv_current": r.hrv_current,
                "spo2_current": r.spo2_current,
                "apnea_now": r.apnea_now,
                "gru_hidden_norm": r.gru_hidden_norm,
                "created_at": r.created_at,
                "state_vector": state_vec,
            })
        return result

    def get_state_at(self, minute: int) -> Optional[dict]:
        """Return twin state at a specific minute."""
        with SessionLocal() as session:
            r = session.query(TwinStateRecord).filter_by(
                subject_id=self.subject_id, minute=minute
            ).first()
        if r is None:
            return None
        state_vec = json.loads(r.state_vector_json) if r.state_vector_json else {}
        return {
            "minute": r.minute,
            "risk_score": r.risk_score,
            "hr_current": r.hr_current,
            "hrv_current": r.hrv_current,
            "spo2_current": r.spo2_current,
            "apnea_now": r.apnea_now,
            "gru_hidden_norm": r.gru_hidden_norm,
            "state_vector": state_vec,
        }

    def clear(self) -> None:
        """Remove all state records for this subject (e.g., start of new night)."""
        with SessionLocal() as session:
            session.query(TwinStateRecord).filter_by(subject_id=self.subject_id).delete()
            session.commit()

    @staticmethod
    def list_subjects() -> list[str]:
        """Return all subject IDs that have stored twin states."""
        with SessionLocal() as session:
            rows = session.query(TwinStateRecord.subject_id).distinct().all()
        return [r[0] for r in rows]


def populate_all_twins(
    features_df: pd.DataFrame,
    model_predict_fn,
    gru_encode_fn=None,
) -> None:
    """
    Populate twin states for all subjects using a trained model.
    Respects temporal ordering (no future leakage) — processes minute-by-minute.

    Args:
        features_df: Full feature matrix from engineer.py
        model_predict_fn: Callable(feature_row_dict) -> float (risk score 0-1)
        gru_encode_fn: Optional callable(signal_window) -> np.ndarray (hidden state)
    """
    subjects = features_df["subject_id"].unique()
    print(f"Populating twin states for {len(subjects)} subjects...")

    for sid in subjects:
        subj_df = features_df[features_df["subject_id"] == sid].sort_values("minute")
        twin = PatientDigitalTwin(subject_id=sid)
        twin.clear()

        for _, row in subj_df.iterrows():
            feat_dict = row.to_dict()
            risk = model_predict_fn(feat_dict)
            gru_h = gru_encode_fn(feat_dict) if gru_encode_fn else None
            twin.update(
                minute=int(row["minute"]),
                feature_row=feat_dict,
                risk_score=float(risk),
                gru_hidden=gru_h,
            )

        print(f"  OK {sid}: {len(subj_df)} twin states stored")
