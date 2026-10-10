# GitHub + Google Colab (Component 3)

**Idea:** the code and the small data files live in GitHub. Colab downloads them, does the heavy work, and you download the results.
The huge raw files (TWOSIDES, OFFSIDES, 14 openFDA label files) are NOT in GitHub. They are only needed for notebook 01 and the label scan of notebook 02, and their results are already saved in `work/`.

## Files in the repo (folder `component3/`)
`c3_utils.py`, `severity.py`, `colab_fp.py`, notebooks `01`–`04`, `02b`, `Colab_full_pipeline.ipynb`, and in `work/`:
`drugs_raw.csv`, `drug_details_raw.csv`, `pairs_raw.csv`, `Y_raw.npy`, `effects.csv`, `smiles_cache.csv`.

## Run it in Colab
1. Open Colab → **File → Open notebook → GitHub tab** → paste `sksivakajan/mac-doc` → open `component3/Colab_full_pipeline.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. In cell A1 the repo name is already set. If the repo is **private**, add a Colab secret named `GITHUB_TOKEN`
   (key icon on the left → Add new secret → paste a GitHub token with *Contents: read* → Notebook access ON).
4. *Runtime → Run all* (30–60 minutes). At the end a zip with the results downloads.

## Never do
* Never paste a token or password into a notebook cell or into GitHub. Use Colab *Secrets* only.
