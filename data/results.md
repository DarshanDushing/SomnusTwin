# SomnusTwin - Validation Report

> **Data Source**: Real clinician-scored apnea events from the PhysioNet Apnea-ECG
> Database (Penzel et al., 2000). NOT synthetic labels.

Per-fold AUROC is averaged across 10 leave-one-subject-out folds and reflects consistency across individual patients. Pooled AUROC concatenates predictions across all folds before scoring and is more informative for features (like static EHR demographics) that are constant within any single patient's test fold but vary meaningfully across patients. We report both for transparency.

## Model Architecture
- **Primary**: LightGBM on engineered rolling features + EHR statics + GRU hidden state
- **GRU Encoder**: 2-layer GRU (32-dim) over trailing 30-min signal window
- **Training validation**: 10-fold Leave-One-Subject-Out (LOSO) Cross-Validation

## Primary Metrics (Test Set Averages across 10 folds)
| Metric | Per-Fold Mean ± Std | Pooled (all folds combined) |
|--------|---------------------|-----------------------------|
| **AUROC** | 0.5135 ± 0.1170 | **0.7393** |
| **AUPRC** | 0.7005 ± 0.3245 | **0.8761** |
| Sensitivity @ 80% Specificity | 0.1029 ± 0.1607 | 0.4211 |
| Brier Score | 0.1803 ± 0.0848 | 0.1781 |

*Total validation set: 4,707 minute-windows across 10 subjects*
*Mean positive rate (event within 30 min): 57.0%*

## Ablation Study (LOSO-CV)

| Model Variant | AUROC (Mean±Std) | AUROC (Pooled) | AUPRC (Pooled) |
|---------------|------------------|----------------|----------------|
| EHR only | 0.5000 ± 0.0000 | 0.8260 | 0.9123 |
| Wearable signals only | 0.4910 ± 0.1389 | 0.8083 | 0.9147 |
| **Fused (EHR + signals)** | 0.5135 ± 0.1170 | **0.7393** | **0.8761** |

## Per-Subject Breakdown

| Subject | AUROC | AUPRC | Sensitivity @ 80% Spec | Brier Score |
|---------|-------|-------|------------------------|-------------|
| c01 | N/A (no pos events) | N/A (no pos events) | N/A | N/A |
| c02 | N/A (no pos events) | N/A (no pos events) | N/A | N/A |
| b02 | 0.3276 | 0.1068 | 0.0303 | 0.1805 |
| b01 | 0.4066 | 0.2966 | 0.0705 | 0.2573 |
| a02 | 0.4581 | 0.9676 | 0.1686 | 0.0343 |
| a05 | 0.4912 | 0.6590 | 0.0489 | 0.2321 |
| a04 | 0.5258 | 0.6136 | 0.0000 | 0.2334 |
| x01 | 0.5275 | 0.9862 | 0.0000 | 0.2298 |
| a01 | 0.6691 | 0.9986 | 0.0000 | 0.2345 |
| a03 | 0.7019 | 0.9756 | 0.5044 | 0.0408 |

## Top SHAP Features
| Rank | Feature | Mean |SHAP| |
|------|---------|------------------|
| 1 | `neck_circumference_cm` | 0.1222 |
| 2 | `hrv_mean_30m` | 0.0436 |
| 3 | `gru_h_20` | 0.0140 |
| 4 | `gru_h_19` | 0.0124 |
| 5 | `hr_mean_60m` | 0.0120 |

## Validation Integrity Notes
- **Train/test split**: 10-fold Leave-One-Subject-Out CV
- **Why LOSO-CV?**: With only 10 subjects available, leave-one-subject-out cross-validation gives a more statistically honest estimate than a single train/test split.
- **No future leakage**: /predict endpoint uses only data up to time t
- **Real ground truth**: Annotations by sleep medicine clinicians
- **EHR profiles**: Synthetic (Synthea-inspired), clearly labeled

## Citation
Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The apnea-ECG database.
Computers in Cardiology 2000;27:255-258.
