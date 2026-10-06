import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.models.predictor import SomnusTwinPredictor
from src.api.main import _get_ehr_row, _load_subject_signal

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
subjects_df = pd.read_parquet(PROCESSED_DIR / "subjects.parquet")
ehr_df = pd.read_parquet(PROCESSED_DIR / "ehr_profiles.parquet")

import src.api.main
src.api.main.subjects_df = subjects_df
src.api.main.ehr_df = ehr_df

predictor = SomnusTwinPredictor.get()
print("Predictor ready:", predictor.is_ready)

print(f"{'Subject ID':<15} | {'Severity':<10} | {'Risk@10m':<10} | {'Risk@120m':<10} | {'Risk@300m':<10}")
print("-" * 65)

for _, row in subjects_df.iterrows():
    sid = row["subject_id"]
    sev = row["osa_severity"]
    ehr_row = _get_ehr_row(sid)
    
    try:
        sig_df = _load_subject_signal(sid)
        max_min = sig_df["minute"].max()
        
        mins = [min(10, max_min), min(120, max_min), min(300, max_min)]
        risks = []
        for m in mins:
            res = predictor.predict_at_minute(sig_df, m, ehr_row)
            risks.append(f"{res['risk_score']:.4f}")
            
        print(f"{sid:<15} | {sev:<10} | {risks[0]:<10} | {risks[1]:<10} | {risks[2]:<10}")
    except Exception as e:
        print(f"{sid:<15} | {sev:<10} | ERROR: {e}")

