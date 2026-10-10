"""Component 3 - accuracy / false-positive-rate tables and thesis figures.

Run in the component3 folder:   python make_thesis_figures.py
Needs the saved models in results/ (made by notebook 04) and the work/ files (made by 02 and 03).
Writes:  results/accuracy_fpr_train_test.csv   and   results/figures/*.png (300 dpi)
"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3_utils as c3

WORK, RESULTS = HERE / "work", HERE / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = c3.SEED
TRAIN_CAP = 60_000
MODELS = ["random_forest", "xgboost", "svm"]
ALL = ["prevalence", "drug_history"] + MODELS
NICE = {"prevalence": "Prevalence (baseline)", "drug_history": "Drug history (baseline)",
        "random_forest": "Random Forest", "xgboost": "XGBoost", "svm": "SVM (linear)"}
COL = {"prevalence": "#9a9a9a", "drug_history": "#6b6b6b", "random_forest": "#2a9d8f",
       "xgboost": "#264fa8", "svm": "#e07a1f"}
SPLITS = {"pair": "Pair split (random pairs)", "unseen_drug": "Unseen-drug split (new medicine)"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)

# ---------------------------------------------------------------- data (same steps as notebook 04)
pairs = pd.read_csv(WORK / "modeling_pairs.csv")
Y = np.load(WORK / "modeling_Y.npy")
effects = pd.read_csv(WORK / "modeling_effects.csv")
drugs = pd.read_csv(WORK / "drugs_final.csv", keep_default_na=False)
z = np.load(WORK / "drug_features.npz")
FP, CM = z["FP"], z["CM"]
ia, ib = pairs["idx_a"].to_numpy(), pairs["idx_b"].to_numpy()
K = Y.shape[1]

def counts(Yt, S, thr):
    P = S >= thr
    T = Yt.astype(bool)
    tp, fp = int((P & T).sum()), int((P & ~T).sum())
    fn, tn = int((~P & T).sum()), int((~P & ~T).sum())
    return tp, fp, fn, tn

def metrics(Yt, S, thr):
    tp, fp, fn, tn = counts(Yt, S, thr)
    n = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(threshold=round(float(thr), 4), accuracy=(tp + tn) / n, false_positive_rate=fp / max(fp + tn, 1),
                recall_TPR=rec, specificity_TNR=tn / max(fp + tn, 1), precision=prec,
                F1=2 * prec * rec / max(prec + rec, 1e-12), balanced_accuracy=(rec + tn / max(fp + tn, 1)) / 2,
                micro_AUROC=roc_auc_score(Yt.ravel(), S.ravel()), micro_AUPRC=average_precision_score(Yt.ravel(), S.ravel()),
                TP=tp, FP=fp, FN=fn, TN=tn)

rows, curves, per_effect, conf = [], {}, {}, {}
for mode in SPLITS:
    log(f"=== {mode}")
    sp = np.load(WORK / f"split_{mode}.npz")
    tr_all, va, te = sp["train"], sp["val"], sp["test"]
    rng = np.random.default_rng(SEED)
    tr = np.sort(rng.choice(tr_all, TRAIN_CAP, replace=False)) if len(tr_all) > TRAIN_CAP else tr_all
    tr = np.sort(np.random.default_rng(SEED + 1).choice(tr, min(20000, len(tr)), replace=False))  # 20k training pairs for the train-set numbers (saves RAM)
    hist = c3.DrugHistory(n_drugs=len(drugs)).fit(ia[tr_all], ib[tr_all], Y[tr_all])

    def build_X(idx, is_train):
        a, b = ia[idx], ib[idx]
        parts = [c3.pair_fp_matrix(a, b, FP), CM[a] | CM[b], CM[a] & CM[b], hist.transform(a, b, Y[idx] if is_train else None)]
        parts = [np.asarray(p_, dtype=np.float32) for p_ in parts]
        return np.concatenate(parts, axis=1)

    XF = {}
    for part, idx, is_tr in (("train", tr, True), ("val", va, False), ("test", te, False)):
        _x = build_X(idx, is_tr); fx = RESULTS / f"_tmpX_{mode}_{part}.npy"; np.save(fx, _x); XF[part] = fx; del _x
    import gc; gc.collect()
    def chunked(fn):
        out = []
        for part in ("train", "val", "test"):
            M = np.load(XF[part], mmap_mode="r")
            out.append(np.vstack([fn(np.asarray(M[i:i + 3000])) for i in range(0, len(M), 3000)]))
            del M
        return tuple(out)
    Ytr, Yva, Yte = Y[tr], Y[va], Y[te]
    scores = {}
    base = c3.PrevalenceBaseline().fit(None, Y[tr_all])
    scores["prevalence"] = chunked(base.predict_proba)
    def hscore(idx, is_train):
        h = hist.transform(ia[idx], ib[idx], Y[idx] if is_train else None)
        return (h[:, :K] + h[:, K:2 * K]) / 2
    scores["drug_history"] = (hscore(tr, True), hscore(va, False), hscore(te, False))
    for name in MODELS:
        f = RESULTS / f"model_{mode}_{name}.joblib"
        model = joblib.load(f)["model"]
        if name == "xgboost":                              # predict on the CPU (avoids GPU memory errors)
            for e in model.estimators_:
                e.get_booster().set_param({"device": "cpu"})
        scores[name] = chunked(lambda X, m=model: c3.predict_scores(m, X))
        log(f"  scored {name}")
        del model; gc.collect()

    for name in ALL:
        Str, Sva, Ste = scores[name]
        thr = c3.best_threshold(Yva, Sva)                  # cut-off chosen on VALIDATION only
        for part, Yt, S in (("train", Ytr, Str), ("validation", Yva, Sva), ("test", Yte, Ste)):
            rows.append({"split": mode, "model": name, "set": part, **metrics(Yt, S, thr)})
        yt, st = Yte.ravel(), Ste.ravel()
        curves[(mode, name)] = (roc_curve(yt, st)[:2], precision_recall_curve(yt, st)[:2],
                                roc_auc_score(yt, st), average_precision_score(yt, st), thr)
        conf[(mode, name)] = counts(Yte, Ste, thr)
        if name in MODELS:
            per_effect[(mode, name)] = np.array([average_precision_score(Yte[:, j], Ste[:, j]) if Yte[:, j].sum() > 0 else np.nan
                                                  for j in range(K)])

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "accuracy_fpr_train_test.csv", index=False)
log("saved results/accuracy_fpr_train_test.csv")

best = {}
for mode in SPLITS:                                        # best model by VALIDATION micro-AUPRC
    d = df[(df.split == mode) & (df.set == "validation") & df.model.isin(MODELS)]
    best[mode] = d.sort_values("micro_AUPRC").iloc[-1].model
log(f"best models: {best}")

# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); plt.close(fig); log("figure " + name)

# 1. model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, (metric, title) in zip(axes, [("micro_AUPRC", "micro-AUPRC (higher is better)"),
                                       ("F1", "micro-F1"), ("false_positive_rate", "False positive rate (lower is better)")]):
    w = 0.38
    for k, mode in enumerate(SPLITS):
        v = [df[(df.split == mode) & (df.set == "test") & (df.model == m)][metric].iloc[0] for m in ALL]
        ax.bar(np.arange(len(ALL)) + (k - 0.5) * w, v, w, color=[COL[m] for m in ALL], alpha=1.0 if k == 0 else 0.55,
               edgecolor="white", hatch="" if k == 0 else "//", label=SPLITS[mode])
        for x, val in zip(np.arange(len(ALL)) + (k - 0.5) * w, v):
            ax.text(x, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(ALL))); ax.set_xticklabels(["Prevalence", "Drug\nhistory", "Random\nForest", "XGBoost", "SVM"], fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylim(0, max(ax.get_ylim()[1], 0.9) if metric != "false_positive_rate" else None)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Test-set comparison of the models and the two baselines", y=1.02, fontsize=13)
save(fig, "fig1_model_comparison")

# 2. ROC curves
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
for ax, mode in zip(axes, SPLITS):
    for m in ALL:
        (fpr, tpr), _, auc, _, _ = curves[(mode, m)]
        ax.plot(fpr, tpr, color=COL[m], lw=2 if m in MODELS else 1.5, ls="-" if m in MODELS else "--", label=f"{NICE[m]}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)"); ax.set_title(SPLITS[mode])
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.suptitle("ROC curves on the test set (all pair-effect cells together)", y=1.01, fontsize=13)
save(fig, "fig2_roc_curves_test")

# 3. precision-recall
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
for ax, mode in zip(axes, SPLITS):
    for m in ALL:
        _, (pr, rc), _, ap, _ = curves[(mode, m)]
        ax.plot(rc, pr, color=COL[m], lw=2 if m in MODELS else 1.5, ls="-" if m in MODELS else "--", label=f"{NICE[m]}  (AUPRC {ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(SPLITS[mode]); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
fig.suptitle("Precision-recall curves on the test set", y=1.01, fontsize=13)
save(fig, "fig3_precision_recall_test")

# 4. confusion matrices of the best model
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, mode in zip(axes, SPLITS):
    tp, fp, fn, tn = conf[(mode, best[mode])]
    M = np.array([[tn, fp], [fn, tp]]); R = M / M.sum(1, keepdims=True)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n({100 * R[i, j]:.1f}%)", ha="center", va="center", fontsize=11,
                    color="white" if R[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nno effect", "Predicted\neffect"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Really\nno effect", "Really\neffect"])
    ax.set_title(f"{SPLITS[mode]}\n{NICE[best[mode]]}", fontsize=11)
    ax.spines[:].set_visible(False)
fig.suptitle("Confusion matrix on the test set (every pair x side-effect cell; % of each real row)", y=1.03, fontsize=12.5)
save(fig, "fig4_confusion_matrix_best_test")

# 5. train vs test
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
for r, mode in enumerate(SPLITS):
    for c, (metric, title) in enumerate([("accuracy", "Accuracy"), ("false_positive_rate", "False positive rate")]):
        ax = axes[r, c]
        for k, part in enumerate(["train", "test"]):
            v = [df[(df.split == mode) & (df.set == part) & (df.model == m)][metric].iloc[0] for m in ALL]
            ax.bar(np.arange(len(ALL)) + (k - 0.5) * 0.38, v, 0.38, color=["#2f6fb3" if part == "train" else "#e0782b"] * len(ALL),
                   label=part.capitalize() + " set")
            for x, val in zip(np.arange(len(ALL)) + (k - 0.5) * 0.38, v):
                ax.text(x, val + 0.008, f"{val:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(ALL))); ax.set_xticklabels(["Prevalence", "Drug history", "RF", "XGBoost", "SVM"], fontsize=9)
        ax.set_title(f"{title} - {SPLITS[mode]}", fontsize=10.5)
        ax.set_ylim(0, 1.0 if metric == "accuracy" else None)
axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Training result versus testing result", y=1.0, fontsize=13)
fig.tight_layout()
save(fig, "fig5_train_vs_test")

# 6. per side effect (best model, pair split)
mode = "pair"
v = per_effect[(mode, best[mode])]
order = np.argsort(np.nan_to_num(v, nan=-1))
sevcol = {"High": "#c0392b", "Medium": "#e6a100", "Low": "#3a9d5d", "Unknown": "#9a9a9a"}
fig, ax = plt.subplots(figsize=(8, 15))
names = effects["effect_name"].to_numpy()
sev = effects["severity"].fillna("Unknown").to_numpy()
ax.barh(range(K), v[order], color=[sevcol.get(s, "#9a9a9a") for s in sev[order]])
ax.set_yticks(range(K)); ax.set_yticklabels(names[order], fontsize=7.5)
ax.set_xlabel("AUPRC on the test set"); ax.set_title(f"Quality for each of the {K} side effects - {NICE[best[mode]]}, {SPLITS[mode]}\n(colour = severity level from the rule table)", fontsize=11)
for s, c in sevcol.items():
    ax.barh([], [], color=c, label=s)
ax.legend(frameon=False, loc="lower right", fontsize=9)
save(fig, "fig6_auprc_per_side_effect")

# 7. pipeline diagram
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))

fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
box(ax, 0.2, 3.6, 2.4, 1.1, "TWOSIDES / OFFSIDES\n(FAERS reports)\n178,982 drug pairs", "#e8eef7")
box(ax, 3.2, 3.6, 2.4, 1.1, "Clean + filter\n70 side effects\n1,807 drugs", "#e8eef7")
box(ax, 6.2, 3.6, 2.4, 1.1, "PubChem structure\n(SMILES) + RDKit\nMorgan fingerprint\n1,024 bits", "#fdf0d5")
box(ax, 9.2, 3.6, 3.5, 1.1, "Pair features\nfingerprints (A or B, A and B)\n+ drug classes + drug history", "#fdf0d5")
box(ax, 0.2, 1.7, 2.4, 1.1, "Split 70 / 20 / 10\ntrain / validation / test\n(pair and unseen-drug)", "#e3f2e8")
box(ax, 3.2, 1.7, 2.4, 1.1, "Models\nRandom Forest\nXGBoost (GPU)\nSVM", "#e3f2e8")
box(ax, 6.2, 1.7, 2.4, 1.1, "Pick cut-off on\nvalidation set,\nscore on test set", "#e3f2e8")
box(ax, 9.2, 1.7, 3.5, 1.1, "Output for a pair\nside-effect probabilities\n+ Low / Medium / High level", "#fbe3e0")
box(ax, 3.2, 0.1, 9.5, 0.9, "Checks: no pair in two parts, no duplicate pairs, label-shuffle test, baselines (prevalence, drug history)", "#f2f2f2")
for x1, x2 in [(2.6, 3.2), (5.6, 6.2), (8.6, 9.2)]:
    arrow(ax, x1, 4.15, x2, 4.15)
arrow(ax, 10.95, 3.6, 1.4, 2.8)
for x1, x2 in [(2.6, 3.2), (5.6, 6.2), (8.6, 9.2)]:
    arrow(ax, x1, 2.25, x2, 2.25)
ax.set_title("Component 3 - method overview", fontsize=13)
save(fig, "fig7_method_overview")

# 8. split diagram
fig, axes = plt.subplots(2, 1, figsize=(11, 4.2))
for ax, mode in zip(axes, SPLITS):
    sp = np.load(WORK / f"split_{mode}.npz"); n = [len(sp["train"]), len(sp["val"]), len(sp["test"])]
    left = 0
    for lab, c, v in zip(["Train", "Validation", "Test"], ["#2f6fb3", "#e6a100", "#c0392b"], n):
        ax.barh(0, v, left=left, color=c, height=0.55)
        ax.text(left + v / 2, 0, f"{lab}\n{v:,} ({100 * v / sum(n):.0f}%)", ha="center", va="center", color="white", fontsize=9.5)
        left += v
    ax.set_title(SPLITS[mode], fontsize=11, loc="left"); ax.axis("off")
fig.suptitle("How the 150,458 drug pairs are divided", fontsize=12.5)
save(fig, "fig8_data_split")

log("DONE. Figures are in " + str(FIG))
