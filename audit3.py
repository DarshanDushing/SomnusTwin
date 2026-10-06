import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path("c:/Users/Darshan/OneDrive/Desktop/Twin")))
from src.models.predictor import SomnusTwinPredictor

processed_dir = Path("data/processed")
subjects_df = pd.read_parquet(processed_dir / "subjects.parquet")
ehr_df = pd.read_parquet(processed_dir / "ehr_profiles.parquet")
predictor = SomnusTwinPredictor.get()

summary = []

for sid in subjects_df['subject_id']:
    sig_df = pd.read_parquet(processed_dir / f"signals_{sid}.parquet")
    ehr_row = ehr_df[ehr_df['subject_id'] == sid].iloc[0] if not ehr_df[ehr_df['subject_id'] == sid].empty else None
    
    pred = predictor.predict_at_minute(sig_df, 100, ehr_row) if 100 <= sig_df['minute'].max() else None
    top_shap = pred['reasons'][0]['feature'] if pred and pred.get('reasons') else "None"
    
    # Fast mean risk approximation using heuristic (since SHAP is too slow to loop)
    # Just to get a rough mean risk across night if not in DB
    mean_risk = pred['risk_score'] if pred else 0.0
    ahi = subjects_df[subjects_df['subject_id'] == sid].iloc[0]['ahi_approx']
    
    summary.append({'subject_id': sid, 'ahi': ahi, 'mean_risk': mean_risk, 'top_shap': top_shap})

print(f"{'Subject':<10} | {'AHI':<6} | {'Risk @ t=100':<12} | {'Top SHAP Reason'}")
print("-" * 60)
for s in summary:
    print(f"{s['subject_id']:<10} | {s['ahi']:<6.1f} | {s['mean_risk']:<12.3f} | {s['top_shap']}")

