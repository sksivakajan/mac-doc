"""Run notebooks 02, 03 and 04 (pair split, then unseen-drug split) on this PC. XGBoost uses the GPU.
Every executed notebook (with all its printed output and plots) is saved in results/executed/.
Usage (in this folder):  python run_pc_all.py            (all steps)
                         python run_pc_all.py 04_pair    (one step: 02, 03, 04_pair, 04_unseen_drug)
"""
import sys, time, json, copy, traceback
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "executed"
OUT.mkdir(parents=True, exist_ok=True)
LOG = HERE / "results" / "pc_run_log.txt"

def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

CPU_PREDICT = ("    trained[name] = model\n",
               "    if name == 'xgboost':      # train on the GPU, but predict on the CPU (avoids GPU out-of-memory)\n"
               "        for _e in model.estimators_:\n"
               "            _e.get_booster().set_param({'device': 'cpu'})\n"
               "    trained[name] = model\n")

def run(nb_path, out_name, edits=None):
    nb = nbformat.read(nb_path, as_version=4)
    # SVM: 12 parallel workers each copy the data as float64 -> out of memory on a 16 GB PC. Use 3 workers.
    SVM_JOBS = ("return OneVsRestClassifier(base, n_jobs=N_JOBS)", "return OneVsRestClassifier(base, n_jobs=2)")
    edits = ([CPU_PREDICT, SVM_JOBS] + list(edits)) if edits else edits
    if edits:
        for c in nb.cells:
            if c.cell_type == "code":
                for old, new in edits:
                    c.source = c.source.replace(old, new)
    t0 = time.time()
    log(f"START {out_name}")
    ep = ExecutePreprocessor(timeout=-1, kernel_name="python3")
    err = None
    try:
        ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
    except Exception as e:
        err = e
        log(f"ERROR in {out_name}: {str(e)[-600:]}")
    nbformat.write(nb, OUT / out_name)
    log(f"{'FAILED' if err else 'DONE'} {out_name} in {(time.time()-t0)/60:.1f} min")
    return err is None

steps = {
    "02": lambda: run(HERE / "02_drug_details_and_features.ipynb", "02_done.ipynb"),
    "03": lambda: run(HERE / "03_split_and_check.ipynb", "03_done.ipynb"),
    "04_pair": lambda: run(HERE / "04_train_xgboost_rf_svm.ipynb", "04_pair_done.ipynb",
                           [('XGB_DEVICE = "cpu"', 'XGB_DEVICE = "cuda"'), ('SPLIT_MODE = "pair"', 'SPLIT_MODE = "pair"')]),
    "04_unseen_drug": lambda: run(HERE / "04_train_xgboost_rf_svm.ipynb", "04_unseen_drug_done.ipynb",
                                  [('XGB_DEVICE = "cpu"', 'XGB_DEVICE = "cuda"'), ('SPLIT_MODE = "pair"', 'SPLIT_MODE = "unseen_drug"')]),
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(steps)
    for k in want:
        if not steps[k]():
            log("Stopping: fix the error above, then run again with the failed step name.")
            break
    log("ALL DONE" if True else "")
