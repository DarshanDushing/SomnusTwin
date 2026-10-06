"""
generate_ehr.py
---------------
Generates clinically plausible synthetic EHR profiles (Synthea-inspired)
for each real PhysioNet Apnea-ECG subject, matched by OSA severity band.

The linkage is explicitly SYNTHETIC — real signal, synthetic demographics.
This is documented honestly in all outputs and the README.

Outputs:
  data/processed/ehr_profiles.parquet  – one row per subject
  data/processed/ehr_profiles.csv

Usage:
  python data/generate_ehr.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "processed"
SEED = 42
rng = np.random.default_rng(SEED)

# ─── Clinically plausible distributions by OSA severity ───────────────────────
# Based on published population statistics for OSA cohorts (cited in README)
SEVERITY_PROFILES = {
    "normal": {
        "age_mean": 42, "age_std": 12,
        "bmi_mean": 24.5, "bmi_std": 3.5,
        "neck_mean": 37.0, "neck_std": 2.5,
        "hypertension_prob": 0.18,
        "sedative_prob": 0.08,
        "alcohol_prob": 0.20,
        "prior_ahi_mean": 2.5, "prior_ahi_std": 1.5,
    },
    "mild": {
        "age_mean": 48, "age_std": 11,
        "bmi_mean": 27.5, "bmi_std": 4.0,
        "neck_mean": 39.5, "neck_std": 2.8,
        "hypertension_prob": 0.32,
        "sedative_prob": 0.12,
        "alcohol_prob": 0.28,
        "prior_ahi_mean": 10.0, "prior_ahi_std": 3.0,
    },
    "moderate": {
        "age_mean": 52, "age_std": 10,
        "bmi_mean": 31.0, "bmi_std": 5.0,
        "neck_mean": 41.5, "neck_std": 3.2,
        "hypertension_prob": 0.52,
        "sedative_prob": 0.18,
        "alcohol_prob": 0.32,
        "prior_ahi_mean": 22.0, "prior_ahi_std": 4.5,
    },
    "severe": {
        "age_mean": 55, "age_std": 10,
        "bmi_mean": 35.5, "bmi_std": 6.0,
        "neck_mean": 44.0, "neck_std": 3.8,
        "hypertension_prob": 0.72,
        "sedative_prob": 0.25,
        "alcohol_prob": 0.38,
        "prior_ahi_mean": 45.0, "prior_ahi_std": 12.0,
    },
}

MALE_PROB = 0.65  # OSA is more prevalent in males


def generate_ehr_for_subject(subject_id: str, severity: str, ahi_approx: float) -> dict:
    """Generate one synthetic EHR profile matched to OSA severity band."""
    p = SEVERITY_PROFILES[severity]

    sex = "M" if rng.random() < MALE_PROB else "F"
    age = max(18, min(85, int(rng.normal(p["age_mean"], p["age_std"]))))
    bmi = max(18.0, round(float(rng.normal(p["bmi_mean"], p["bmi_std"])), 1))
    neck_cm = max(30.0, round(float(rng.normal(p["neck_mean"], p["neck_std"])), 1))

    # Adjust neck for sex (males typically larger)
    if sex == "M":
        neck_cm += 2.0

    hypertension = bool(rng.random() < p["hypertension_prob"])
    sedative_use = bool(rng.random() < p["sedative_prob"])
    alcohol_use = bool(rng.random() < p["alcohol_prob"])

    # Prior AHI from a previous sleep study (± variability)
    prior_ahi = max(0.0, round(float(rng.normal(p["prior_ahi_mean"], p["prior_ahi_std"])), 1))

    # Epworth Sleepiness Scale (0-24), higher in more severe OSA
    ess_base = {"normal": 6, "mild": 9, "moderate": 13, "severe": 17}[severity]
    ess_score = max(0, min(24, int(rng.normal(ess_base, 3))))

    # STOP-BANG score component flags (synthetic, consistent with profile)
    stopbang_snoring = severity in ("moderate", "severe") or rng.random() < 0.5
    stopbang_tired = ess_score > 10
    stopbang_observed = severity in ("moderate", "severe")
    stopbang_pressure = hypertension
    stopbang_bmi = bmi > 35
    stopbang_age = age > 50
    stopbang_neck = neck_cm > 40
    stopbang_male = sex == "M"
    stopbang_total = sum([
        stopbang_snoring, stopbang_tired, stopbang_observed, stopbang_pressure,
        stopbang_bmi, stopbang_age, stopbang_neck, stopbang_male
    ])

    # CPAP-prescribed if moderate/severe
    cpap_prescribed = severity in ("moderate", "severe") and rng.random() < 0.65
    cpap_adherent = cpap_prescribed and rng.random() < 0.60

    return {
        "subject_id": subject_id,
        # Demographics
        "sex": sex,
        "age": age,
        "bmi": bmi,
        "neck_circumference_cm": neck_cm,
        # Clinical flags
        "hypertension": int(hypertension),
        "sedative_use": int(sedative_use),
        "alcohol_use": int(alcohol_use),
        # Sleep history
        "prior_ahi": prior_ahi,
        "osa_severity": severity,
        "ahi_real": ahi_approx,
        "ess_score": ess_score,
        # STOP-BANG
        "stopbang_score": stopbang_total,
        # Treatment
        "cpap_prescribed": int(cpap_prescribed),
        "cpap_adherent": int(cpap_adherent),
        # Metadata
        "synthetic_linkage": True,   # always honest about this
        "linkage_basis": "ahi_severity_band",
    }


def main():
    subjects_path = PROCESSED_DIR / "subjects.parquet"
    if not subjects_path.exists():
        raise FileNotFoundError(
            "subjects.parquet not found. Run fetch_physionet.py first."
        )

    subjects_df = pd.read_parquet(subjects_path)
    print(f"Generating EHR profiles for {len(subjects_df)} subjects...")

    ehr_rows = []
    for _, row in subjects_df.iterrows():
        ehr = generate_ehr_for_subject(
            subject_id=row["subject_id"],
            severity=row["osa_severity"],
            ahi_approx=row["ahi_approx"],
        )
        ehr_rows.append(ehr)

    ehr_df = pd.DataFrame(ehr_rows)
    ehr_df.to_parquet(PROCESSED_DIR / "ehr_profiles.parquet", index=False)
    ehr_df.to_csv(PROCESSED_DIR / "ehr_profiles.csv", index=False)

    # Print summary
    print("\nEHR Profile Summary:")
    print(f"  Subjects: {len(ehr_df)}")
    print(f"  Sex distribution: {ehr_df['sex'].value_counts().to_dict()}")
    print(f"  Age: {ehr_df['age'].mean():.1f} ± {ehr_df['age'].std():.1f} years")
    print(f"  BMI: {ehr_df['bmi'].mean():.1f} ± {ehr_df['bmi'].std():.1f}")
    print(f"  Hypertension prevalence: {ehr_df['hypertension'].mean():.1%}")
    print(f"  Sedative use: {ehr_df['sedative_use'].mean():.1%}")
    print(f"  CPAP prescribed: {ehr_df['cpap_prescribed'].mean():.1%}")

    print(f"\nOK Saved to {PROCESSED_DIR}/ehr_profiles.parquet")
    print("\nIMPORTANT: EHR profiles are SYNTHETIC, matched by OSA severity band.")
    print("Signal data (ECG, annotations) are REAL PhysioNet Apnea-ECG recordings.")

    # Also save a combined summary JSON for the dashboard "About" panel
    summary = {
        "data_sources": {
            "signals": "PhysioNet Apnea-ECG Database (real, expert-annotated)",
            "ehr_profiles": "Synthea-inspired synthetic profiles (honest synthetic linkage)",
            "linkage_basis": "OSA severity band (AHI-derived from real annotations)",
        },
        "n_subjects": len(ehr_df),
        "severity_distribution": ehr_df["osa_severity"].value_counts().to_dict(),
        "disclaimer": (
            "EHR demographic profiles are synthetic. "
            "Physiological signals and apnea annotations are real, "
            "sourced from PhysioNet Apnea-ECG Database (Penzel et al., 2000)."
        ),
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
