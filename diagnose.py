import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path("c:/Users/Darshan/OneDrive/Desktop/Twin")))
from src.models.predictor import SomnusTwinPredictor

def run_diagnostics():
    processed_dir = Path("data/processed")
    subjects_df = pd.read_parquet(processed_dir / "subjects.parquet")
    ehr_df = pd.read_parquet(processed_dir / "ehr_profiles.parquet")
    predictor = SomnusTwinPredictor.get()
    
    # 1. Within-subject variation test
    print("=== WITHIN-SUBJECT VARIATION TEST ===")
    test_subjects = ["a01", "c01"] # a01 is severe, c01 is mild
    for sid in test_subjects:
        sig_df = pd.read_parquet(processed_dir / f"signals_{sid}.parquet")
        ehr_row = ehr_df[ehr_df['subject_id'] == sid].iloc[0]
        
        max_t = sig_df['minute'].max()
        times = np.linspace(10, max_t - 10, 20, dtype=int)
        
        risks = []
        for t in times:
            p = predictor.predict_at_minute(sig_df, t, ehr_row)
            risks.append(p['risk_score'])
            
        print(f"\nSubject {sid} (AHI: {subjects_df[subjects_df['subject_id']==sid]['ahi_approx'].iloc[0]})")
        print(f"Risk over 20 points: {['%.3f' % r for r in risks]}")
        print(f"Variance: {np.var(risks):.4f}")

    # 2. Feature Importance Check
    print("\n=== FEATURE IMPORTANCE CHECK ===")
    import json
    with open("data/models/feature_names.json") as f:
        config = json.load(f)
    
    model = predictor.model
    # LightGBM feature importances
    importances = model.feature_importances_
    features = config["all_feature_names"]
    
    df_imp = pd.DataFrame({'feature': features, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False).head(10)
    print("Top 10 features by LightGBM split importance:")
    for _, row in df_imp.iterrows():
        print(f"  {row['feature']}: {row['importance']}")

if __name__ == '__main__':
    run_diagnostics()
