import requests
import numpy as np

def audit():
    try:
        resp = requests.get('http://localhost:8000/subjects')
        subjects = resp.json().get('subjects', [])
    except Exception as e:
        print("Could not fetch subjects. Ensure the server is running on port 8000.", e)
        return

    hr_arrays = {}
    summary = []

    for sub in subjects:
        sid = sub['subject_id']
        night_resp = requests.get(f'http://localhost:8000/subjects/{sid}/night').json()
        predict_resp = requests.get(f'http://localhost:8000/subjects/{sid}/predict?t=100').json()
        twin_resp = requests.get(f'http://localhost:8000/subjects/{sid}/twin-state').json()
        
        signal = night_resp.get('signal', [])
        hr_array = [s.get('hr', 0) or 0 for s in signal]
        hr_arrays[sid] = hr_array
        
        reasons = predict_resp.get('reasons', [])
        top_reason = reasons[0].get('feature', 'None') if reasons else 'None'
        
        trajectory = twin_resp.get('trajectory', [])
        risks = [s.get('risk_score', 0) for s in trajectory]
        mean_risk = np.mean(risks) if risks else 0
        
        summary.append({
            'subject_id': sid,
            'ahi': sub.get('ahi_approx', 0),
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
