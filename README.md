# SomnusTwin
### An explainable Digital Twin for early sleep apnea risk prediction

> Built for the Happiest Health **Digital Twin Challenge 2026**

---

## 1. Team & Institute

| Role | Name | Institute |
|---|---|---|
| Team Leader | Darshan Dushing | DJSCE |
| Member 2 | Janvhi Vavre | Hitech Institute of Technology |

**Repo folder name on submission:** `SomnusTwin`

---

## 2. Project Title

**SomnusTwin — An Explainable Digital Twin for Early Obstructive Sleep Apnea (OSA) Risk Prediction**

---

## 3. The Problem We're Trying to Solve

Obstructive Sleep Apnea is common in India and badly under-diagnosed. Most people with it never see a sleep lab — awareness is low, overnight polysomnography is expensive and only available in a handful of cities, and the condition quietly raises the risk of hypertension, stroke, and cardiac arrhythmia in the background. By the time someone finds out they have it, it's often because something worse has already happened.

We picked OSA over the organizer's worked examples (diabetes, hypertension) for a specific reason: it's one of the few chronic conditions where genuinely open, clinician-labeled physiological data exists — the PhysioNet Apnea-ECG Database. That let us validate this prototype against real expert annotations instead of leaning entirely on synthetic labels, which is hard to do for most other conditions in the time we had.

### What we're actually predicting
> The probability that a patient will have a significant apnea or desaturation event in the **next 30 minutes**, computed continuously from their clinical history and real-time signal trends — with a plain-language explanation of *why* the model thinks so, right now.

### Who this is for
Sleep medicine clinicians and technicians reviewing overnight recordings, or managing CPAP titration for diagnosed patients — a tool that flags rising risk *before* a desaturation event, instead of only showing it in hindsight.

---

## 4. Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| ML / Data | pandas, numpy, scikit-learn, LightGBM, PyTorch (GRU encoder), SHAP |
| Digital Twin state | SQLite — persisted, timestamped, queryable per subject per minute |
| Frontend | React, Vite, TailwindCSS |
| Data | PhysioNet Apnea-ECG Database (real) + Synthea-inspired synthetic EHR |

---

## 5. How the Model Works

### The data
- **Signals (real):** PhysioNet Apnea-ECG Database — Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. *The Apnea-ECG Database*, Computers in Cardiology 2000;27:255-258. Continuous ECG with per-minute expert apnea/normal annotations, fully open access. A subset of subjects also has a paired SpO2 channel.
- **EHR profiles (synthetic):** Age, sex, BMI, neck circumference, and clinical history, generated independently of that night's signal — deliberately *not* derived from the subject's own apnea count (more on why that matters below).

### Feature engineering
Per-minute HR and HRV extracted from raw ECG, with SpO2 used directly where available. On top of that, rolling-window features at 5/15/30/60-minute horizons — mean, standard deviation, and trend — for HR, HRV, and SpO2, plus explicit missingness flags so the model (and the dashboard) can represent lower confidence when a sensor channel drops out, instead of silently guessing.

### The Digital Twin state
Every subject has a persisted, timestamped state vector, written to SQLite and updated every minute: their static profile plus current rolling signal features plus a GRU-encoded summary of the trailing raw window. You can pull up any subject's full night and watch this state evolve minute by minute — that's what makes this a twin rather than a one-shot classifier.

### The model itself
A 2-layer GRU compresses the trailing raw HR/HRV/SpO2 sequence into a hidden state, capturing temporal patterns the hand-built features miss. LightGBM then takes the engineered features + safe demographics + GRU hidden state and predicts event-within-30-minutes. SHAP (TreeExplainer) computes real per-prediction attributions, which the dashboard turns into plain-language reason codes.

### How we validated it
With only 10 subjects available from the open dataset, a single train/test split would be close to meaningless — one lucky or unlucky split could swing the numbers wildly. So we used **leave-one-subject-out cross-validation**: 10 folds, each training on 9 subjects and testing on the one held out, with no subject ever appearing in both. Every prediction also uses only signal data up to and including that exact minute — nothing from the future leaks in.

### Results — and why we're reporting two different numbers

| Metric | Per-fold (mean ± std) | Pooled (all folds combined) |
|---|---|---|
| AUROC — Fused | 0.514 ± 0.117 | **0.739** |
| AUROC — EHR only | 0.500 ± 0.000 | 0.829 |
| AUROC — Wearable only | 0.564 ± 0.131 | 0.879 |
| AUPRC — Fused | — | 0.876 |

*4,707 minute-windows across 10 subjects, after the feature-pipeline fixes described below. These are our final, post-fix numbers.*

These two numbers answer genuinely different questions, and we think reporting both — rather than picking whichever looks better — is the more honest way to present this:

- **Pooled AUROC** asks: *across all patients, can the model rank who's higher-risk overall?* This is driven mostly by static history — EHR alone already gets to 0.83 here, because age, BMI, and clinical background vary a lot *between* people. This is closer to the kind of triage a clinician could already approximate from intake notes.
- **Per-fold AUROC** asks the harder question: *for one specific, already-known patient, can the model tell when risk is rising right now, tonight?* This is the real job of a Digital Twin, and it's where EHR alone is flat at 0.50 (because a patient's age and BMI don't change minute to minute — there's nothing to detect), while wearable signal alone reaches 0.564. We treat this as our primary metric, because it reflects the deployed use case: watching one patient over time, not sorting a population.

We'll say the obvious part out loud: these are modest numbers for a hard, short-horizon prediction task on ten subjects. We'd rather show that honestly than round up.

### The debugging story (we're keeping this in, on purpose)

We thought about cleaning this section out and just presenting the final numbers. We decided not to — how we got here says more about whether you can trust this model than the numbers alone do.

It started when an early version of the model scored 0.9998 AUROC. That's not a good sign, it's a red flag — nothing about predicting a sleep apnea event thirty minutes ahead should be that easy. So we didn't celebrate it, we went looking for what was wrong. Turned out the model had found a shortcut: one of its features was quietly built from the same night's ground-truth labels it was supposed to be predicting. It wasn't cheating on purpose, obviously, but that's exactly what it was doing — reading the answer key. We caught it the simple way: we checked whether the model's predicted risk ever actually moved during a single patient's night. It didn't. It just sat at 0% or 100% for hours, which is not what a model paying attention to a heartbeat should do. We pulled the leaky features out and started over.

The second bug was sneakier, and honestly a bit worse. Two of our patients only had heart data, no oxygen-saturation sensor — a real limitation of the open dataset we used, not something we could fix. The problem was in how we handled that gap: we were filling the missing value with zero, which sounds harmless until you remember that 0% blood oxygen is not a "missing reading," it's a number that would mean the patient is in cardiac arrest. The model had never seen anything close to that during training, so it did something strange with it — it read that impossible zero as "suspiciously perfect," and quietly pushed its risk estimate toward zero, even during a real, clinician-confirmed apnea event happening at that exact moment. We only found this because we went looking minute-by-minute at one patient's night and noticed the model was practically asleep itself while the patient was mid-event. The fix was straightforward once we saw it: a missing reading should look "average," not "impossible," so we now fill gaps with the training set's typical value instead of zero.

The third one was just a mistake, and we're happy to say so. While we were stripping out the leaky features from bug one, a perfectly innocent feature — how long it's been since the patient's last apnea event — got swept out along with them, because its name happened to look similar. We found it by literally listing every feature our pipeline builds and comparing it against every feature the model was actually allowed to see. It's back in now.

There's one more thing worth mentioning, not because it's a bug, but because we checked it carefully enough to be sure it *isn't* one. For our most severe patients, the predicted risk sometimes holds at exactly the same number for a long stretch of the night — not approximately the same, identical down to six decimal places. We double-checked the raw numbers specifically because that pattern made us suspicious. It turned out to be a real, explainable property of how the model works: decision trees only change their output when a value crosses a specific threshold they learned during training, and for a patient whose breathing is disrupted almost continuously, the small ups and downs in heart rate between events often just aren't big enough to cross one. We're telling you this plainly rather than hiding it: for patients with very frequent apnea (more than 90% of the night) and no oxygen sensor, the model's sense of "right now" gets coarser, and it settles into a steady, elevated estimate instead of tracking every small fluctuation. More patients in the training data, and a little less regularization, would likely sharpen this — we simply ran out of time and data to chase it further without risking undoing the leakage fix we'd just made.

---

## 6. Running It Yourself

```
Real PhysioNet ECG/SpO2 + Synthetic EHR
            ↓
  Feature engineering (rolling windows, missingness flags)
            ↓
  GRU encoder (temporal) ──┐
                             ├─→ Digital Twin State (SQLite, persisted)
  Static clinical features ─┘
            ↓
      LightGBM classifier
            ↓
     SHAP explainability
            ↓
   Doctor-facing dashboard
```

```bash
# Backend
pip install -r requirements.txt
python data/fetch_physionet.py        # real PhysioNet data (needs internet)
python data/generate_ehr.py           # synthetic EHR, independent of labels
python src/features/engineer.py
python src/twin/state.py
python src/models/train.py            # LOSO-CV training + evaluation
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend, separate terminal
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Or just `python main.py` to launch both together.

---

## 7. Demo Video
**[LINK TO BE ADDED]**

## 8. License
MIT — see [LICENSE](LICENSE).

## 9. Architecture Diagram
See [View Architecture Diagram](docs/architecture_diagram.pdf).

## 10. Presentation
See [View Presentation]("docs/SomnusTwin_Presentation.pptx").

---

## Where We Sit Next to Existing Work

| | Twin Health | Biofourmis | SomnusTwin |
|---|---|---|---|
| Domain | Metabolic, whole-body | Remote monitoring, general | OSA, one condition |
| Horizon | Months (lifestyle coaching) | Deviation from baseline | **30 minutes, acute event** |
| Ground truth | Proprietary trial data | Proprietary | **Open, clinician-annotated** |
| Explainability | Not public | Not public | **Exact SHAP, shown live** |
| Validation reported | Clinical trial outcomes | Not public | **Leave-one-subject-out CV, reported honestly either way** |

We're not going to pretend we're competing with platforms built on years of proprietary clinical-trial data — we don't have that data, and honestly, we wouldn't trust a ten-day comparison against it even if we did. What we're offering instead is narrower and, we think, more honest: a short-horizon, fully explainable, fully reproducible proof-of-concept, built entirely on data anyone can go download themselves, and validated as carefully as a ten-subject student project reasonably can be.

---

## What This Is, and Isn't

Let's be plain about this, because it matters more than any metric above. This is a hackathon proof-of-concept, built on open and synthetic data, by a two-person team, in a few weeks. It hasn't been clinically validated. It isn't a medical device. It does not diagnose or treat anyone, and it was never meant to. Every risk score and recommendation this tool produces is there to support a qualified clinician's judgment — not to replace it, and definitely not to act on its own. The dataset behind it is small, just 10 subjects from an open research database, so please read everything above as a demonstration that this *kind* of system can work, not as a performance number you'd want riding on someone's actual care.
