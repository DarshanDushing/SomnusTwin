import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.predictor import SomnusTwinPredictor

def audit():
    processed_dir = Path("data/processed")
    subjects_df = pd.read_parquet(processed_dir / "subjects.parquet")
    ehr_df = pd.read_parquet(processed_dir / "ehr_profiles.parquet")
    
    predictor = SomnusTwinPredictor.get()
    
    hr_arrays = {}
    summary = []
    
    for sid in subjects_df['subject_id']:
        sig_df = pd.read_parquet(processed_dir / f"signals_{sid}.parquet")
        hr_arrays[sid] = sig_df['hr_bpm'].fillna(0).values
        
        ehr_row = ehr_df[ehr_df['subject_id'] == sid].iloc[0] if not ehr_df[ehr_df['subject_id'] == sid].empty else None
        
        # Predict at t=100
        pred = predictor.predict_at_minute(sig_df, 100, ehr_row) if 100 <= sig_df['minute'].max() else None
        reasons = pred['reasons'] if pred else []
        top_reason = reasons[0]['feature'] if reasons else "None"
        
        # Sample risks to get mean
        risks = []
        for t in range(0, sig_df['minute'].max(), 30):
            p = predictor.predict_at_minute(sig_df, t, ehr_row)
            risks.append(p['risk_score'])
            
        mean_risk = np.mean(risks) if risks else 0
        ahi = subjects_df[subjects_df['subject_id'] == sid].iloc[0]['ahi_approx']
        
        summary.append({
            'subject_id': sid,
            'ahi': ahi,
            'mean_risk': mean_risk,
            'top_shap': top_reason
        })

    print(f"{'Subject':<10} | {'AHI':<6} | {'Mean Risk':<10} | {'Top SHAP Reason'}")
    print("-" * 50)
    for s in summary:
        print(f"{s['subject_id']:<10} | {s['ahi']:<6.1f} | {s['mean_risk']:<10.3f} | {s['top_shap']}")

    print("\n--- Correlation Check ---")
    sids = list(hr_arrays.keys())
    for i in range(len(sids)):
        for j in range(i+1, len(sids)):
            a = hr_arrays[sids[i]]
            b = hr_arrays[sids[j]]
            min_len = min(len(a), len(b))
            if min_len > 0:
                corr = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
                if corr > 0.95:
                    print(f"SUSPICIOUS: {sids[i]} and {sids[j]} have correlation {corr:.3f}")

if __name__ == '__main__':
    audit()
