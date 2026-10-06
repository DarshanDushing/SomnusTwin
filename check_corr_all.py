import pandas as pd
import numpy as np
from itertools import combinations

subjects = pd.read_parquet("data/processed/subjects.parquet")['subject_id'].tolist()
hr_arrays = {}

for s in subjects:
    df = pd.read_parquet(f"data/processed/signals_{s}.parquet")
    hr_arrays[s] = df['hr_bpm'].fillna(0).values

print("Checking all pairs for high correlation:")
for s1, s2 in combinations(subjects, 2):
    min_len = min(len(hr_arrays[s1]), len(hr_arrays[s2]))
    if min_len > 0:
        corr = np.corrcoef(hr_arrays[s1][:min_len], hr_arrays[s2][:min_len])[0,1]
        if corr > 0.95:
            print(f"IDENTICAL FOUND: {s1} and {s2} -> {corr}")

print("Done checking.")
