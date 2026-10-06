import pickle
from pathlib import Path
import json
import numpy as np

MODELS_DIR = Path("data/models")
with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "feature_names.json", "rb") as f:
    config = json.load(f)

importances = model.feature_importances_
total = np.sum(importances)
all_feat_names = config["all_feature_names"]

print("Feature Importances:")
indices = np.argsort(importances)[::-1]
for i in range(10):
    idx = indices[i]
    print(f"{all_feat_names[idx]}: {importances[idx]/total*100:.2f}%")
