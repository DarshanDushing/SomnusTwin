import pandas as pd
import numpy as np

for i in range(1, 4):
    df = pd.read_parquet(f"data/processed/signals_a0{i}.parquet")
    print(f"a0{i} HR (first 5): {df['hr_bpm'].head().values}")

df1 = pd.read_parquet("data/processed/signals_a01.parquet")
df2 = pd.read_parquet("data/processed/signals_a02.parquet")

min_len = min(len(df1), len(df2))
if min_len > 0:
    hr1 = df1['hr_bpm'].fillna(0).values[:min_len]
    hr2 = df2['hr_bpm'].fillna(0).values[:min_len]
    print("Correlation between a01 and a02:", np.corrcoef(hr1, hr2)[0,1])
