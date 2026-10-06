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
    
    # SHAP at t=100
    pred_100 = predictor.predict_at_minute(sig_df, 100, ehr_row) if 100 <= sig_df['minute'].max() else None
    top_shap = pred_100['reasons'][0]['feature'] if pred_100 and pred_100.get('reasons') else "None"
    
    # Calculate mean risk using a few sample points across the night
    risks = []
    for t in range(0, sig_df['minute'].max(), 30):
        # We only need the risk score, but `predict_at_minute` calculates SHAP which is slow.
        # We'll just do 5 points.
        if len(risks) >= 5: break
        p = predictor.predict_at_minute(sig_df, t, ehr_row)
        risks.append(p['risk_score'])
        
    mean_risk = np.mean(risks) if risks else 0.0
    ahi = subjects_df[subjects_df['subject_id'] == sid].iloc[0]['ahi_approx']
    
    summary.append({'subject_id': sid, 'ahi': ahi, 'mean_risk': mean_risk, 'top_shap': top_shap})

print(f"{'Subject':<10} | {'AHI':<6} | {'Mean Risk':<12} | {'Top SHAP Reason'}")
print("-" * 60)
for s in summary:
    print(f"{s['subject_id']:<10} | {s['ahi']:<6.1f} | {s['mean_risk']:<12.3f} | {s['top_shap']}")
