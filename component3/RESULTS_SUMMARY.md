# MedSafe-AI · Component 3 · Results summary (run on 10 Oct 2026)

Task: given two medicines, predict which of 70 side effects may be reported for the pair, and a Low / Medium / High level.
Data: TWOSIDES/OFFSIDES (reports from FAERS). "Not reported" does NOT mean "safe".

## What was done (steps)
1. Cleaned TWOSIDES: 178,982 pairs, 1,807 drugs, 70 side effects (notebook 01).
2. Found each drug's structure on PubChem and made Morgan fingerprints (1024 bits, radius 2) with RDKit in Google Colab, because RDKit is blocked on the PC (Windows Application Control). Result: 1,534 of 1,807 drugs have a fingerprint (84.9%). 172 drugs are not on PubChem (mixtures such as Air, Coal Tar) and 101 got no answer in time.
3. Kept only pairs where both drugs have a fingerprint: 150,458 pairs, 1,528 drugs, 70 side effects (notebook 02).
4. Made the 70 / 20 / 10 split (train / validation / test) in two ways, and checked it (notebook 03):
   - pair split: random pairs. Train 105,321, val 30,092, test 15,045.
   - unseen-drug split: validation and test pairs contain a drug the model never saw in training. Train 104,892, val 30,355, test 15,211.
   - Checks: no pair in two parts, no duplicate pairs. All hard checks passed.
5. Trained Random Forest, XGBoost (on the RTX 4050 GPU) and SVM, and compared with two simple baselines (notebook 04).
   - prevalence = always predict the most common side effects.
   - drug_history = predict from what each drug causes in other training pairs.
   - RF and XGBoost used the first 60,000 training pairs; SVM used 30,000 (time / memory limit).
6. Label-shuffle test (labels mixed up on purpose): both splits PASS (test micro-AUPRC 0.269 and 0.291, below the prevalence baseline 0.310 / 0.314). So there is no sign of leakage.

## Results on the TEST set
| split | model | micro-AUPRC | P@3 | P@10 | micro-F1 | High-effect recall |
|---|---|---|---|---|---|---|
| pair | prevalence | 0.310 | 0.385 | 0.346 | 0.402 | 0.268 |
| pair | drug_history | 0.607 | 0.618 | 0.501 | 0.593 | 0.401 |
| pair | random_forest | 0.755 | 0.620 | 0.511 | 0.675 | 0.627 |
| pair | **xgboost** | **0.768** | 0.617 | 0.511 | **0.682** | **0.675** |
| pair | svm | 0.737 | 0.607 | 0.500 | 0.665 | 0.645 |
| unseen drug | prevalence | 0.314 | 0.366 | 0.354 | 0.413 | 0.261 |
| unseen drug | **drug_history** | **0.449** | **0.514** | **0.436** | **0.494** | 0.333 |
| unseen drug | random_forest | 0.427 | 0.488 | 0.425 | 0.483 | 0.438 |
| unseen drug | xgboost | 0.390 | 0.425 | 0.380 | 0.467 | 0.509 |
| unseen drug | svm | 0.338 | 0.402 | 0.346 | 0.432 | 0.607 |

## How to read it (simple English)
* Pair split: all three models beat both baselines in micro-AUPRC and micro-F1. XGBoost is best (0.768). For the top-3 guesses (P@3) the models are only equal to drug_history (about 0.62).
* Unseen-drug split (a NEW medicine): the models do NOT beat the drug_history baseline (RF 0.427 vs 0.449). They only beat it in High-effect recall (XGBoost 0.509, SVM 0.607 vs 0.333), but with more false alarms. So the fingerprints alone do not yet help enough for a brand-new drug. This is the honest result and it is the one to report.
* Gap between train and test is large (0.88-0.92 train vs 0.39-0.77 test): the models memorise training data.

## Severity (Low / Medium / High) - read with care
Severity comes from placeholder rules in `severity.py`. They are NOT from an agreed medical source. Pair severity accuracy: 0.698 (pair split, XGBoost), 0.678 (unseen drug, RF); High-severity pairs found: 0.793 and 0.847. Do not present these as medical facts until the rule is agreed with the supervisor.

## Limits
* Only 334 of 1,534 drugs have a drug class (so class features are weak).
* 273 drugs (15%) have no fingerprint and were removed.
* FAERS reports show association, not cause.
* RF/XGBoost/SVM did not use all training pairs.

## Where the files are
* `results/metrics_pair.csv`, `metrics_unseen_drug.csv`, `metrics_both_splits.csv`, `comparison_*.png`
* `results/executed/*.ipynb` = every notebook with all printed output and plots (nothing needs to run again)
* `results/pc_run_log.txt` = run log
* `work/drug_fp_colab.npz` = fingerprints (Colab never needs to repeat Step 2b)
* Trained models (`results/*.joblib`, about 3 GB) are only on the PC (too big for GitHub): `D:\med\data set\component3\results`.
* To repeat on the PC: `python run_pc_all.py` (needs a GPU for the fast XGBoost; set in the script).

## Next steps
1. Agree the severity source (e.g. SIDER / drug labels) with the supervisor.
2. Improve the unseen-drug case: more drug features (targets, ATC class, text embeddings), train on all pairs, tune.
3. Add SHAP explanations and the plain-language explanation layer, then the API/app.
