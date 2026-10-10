# Component 3 notebooks – MedSafe-AI

Run the notebooks in this order. Each one saves its results in the `work` folder, so you can stop and continue later.

| # | Notebook | What it does | Time |
|---|---|---|---|
| 01 | `01_prepare_twosides_offsides.ipynb` | Reads TWOSIDES (43 M rows) in pieces, removes bad lines, keeps strong signals (PRR >= 2, A >= 3), builds the pair x side-effect table, and the known side effects from OFFSIDES | 3-6 min |
| 02 | `02_drug_details_and_features.ipynb` | Drug class / use / label side effects from the 14 openFDA files, SMILES from PubChem (internet), Morgan fingerprints (RDKit) | 15-30 min |
| 02b | `02b_fingerprints_colab.ipynb` | **Only if Windows blocks RDKit** (error: *Application Control policy has blocked this file*). Makes Step 4-6 of notebook 02 in Google Colab. Upload 6 small files, download a zip, unzip it into `work` | 5 min |
| 03 | `03_split_and_check.ipynb` | 70 / 20 / 10 split (random pairs AND unseen drugs) + data checks | 1 min |
| 04 | `04_train_xgboost_rf_svm.ipynb` | XGBoost, Random Forest, SVM + 2 baselines, threshold on validation, test once, label-shuffle leakage test, severity | 30-90 min |

## Start
```
cd "D:\med\data set\component3"
pip install pandas numpy matplotlib scikit-learn xgboost joblib rdkit jupyter
jupyter notebook
```

## If notebook 02 stops at Step 3 or Step 4
* Step 3 (PubChem): if it says some drugs are `unknown`, run the cell again. Temporary PubChem/internet problems are never saved as "not found". Only continue when `unknown` is 0.
* Step 4 (RDKit): if the message says Windows is blocking RDKit, do not work around it. Use `02b_fingerprints_colab.ipynb` in Google Colab, unzip the result into `work`, then go on with 03.

## Important settings (first cells)
* 04: `SPLIT_MODE = "pair"` then run again with `"unseen_drug"`. Step 11 puts both results side by side.
* 04: `TRAIN_CAP = 60_000` pairs (set `None` for all ~125 000; needs more RAM and time). `TUNE = True` tries 2 settings per model.
* 01: `PRR_MIN`, `A_MIN`, `TOP_K` change how many pairs and side effects are used. Run 02 and 03 again after you change them.

## Honest limits
* TWOSIDES = statistical signals from adverse-event reports, not proof of cause. "Not listed" does not mean "safe".
* Severity (Low/Medium/High) comes from the placeholder rules in `severity.py`. Agree a real source with the supervisor.
* Only about 30 % of drugs have a drug class in the openFDA labels; the class features are therefore sparse.
* The two baselines (`prevalence`, `drug_history`) must be beaten before a model counts as useful.


## Google Colab + GitHub (new)
* `Colab_full_pipeline.ipynb` does everything in Colab (RDKit works there): fingerprints, split, models. See `GITHUB_COLAB.md`.
* `colab_fp.py` makes `work/drug_fp_colab.npz`. If that file exists, notebook 02 uses it and needs **no RDKit** on your PC.
