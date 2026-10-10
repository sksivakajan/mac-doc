"""MedSafe-AI - Component 3 - drug structures (PubChem) and Morgan fingerprints (RDKit).

Meant for Google Colab (RDKit works there). Needs a table with columns drug_id, drug_name.
Result: drug_fp_colab.npz  (ids, smiles, status, fp, valid, n_bits, radius)
which notebook 02 on your PC can read without RDKit.
"""
import re, json, time, threading, urllib.request, urllib.parse, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

N_BITS, RADIUS = 1024, 2
PUBCHEM = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,CanonicalSMILES/JSON")
UA = {"User-Agent": "MedSafe-AI-student-project/1.0 (SLIIT research)"}

def norm_name(s):
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

class Transient(Exception):
    pass

_lock, _last = threading.Lock(), [0.0]
def _wait(gap=0.25):                      # all threads together: at most 4 requests per second
    with _lock:
        d = gap - (time.time() - _last[0])
        if d > 0:
            time.sleep(d)
        _last[0] = time.time()

def fetch_json(url, retries=5, pause=3.0):
    for attempt in range(retries):
        _wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None               # PubChem really does not know this name
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(pause * (2 ** attempt)); continue
            raise Transient(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(pause * (2 ** attempt))
    raise Transient("busy or no internet")

def parse_smiles(data):
    try:
        props = data["PropertyTable"]["Properties"][0]
    except (TypeError, KeyError, IndexError):
        return None
    keys = [k for k in props if "SMILES" in k.upper()]
    if not keys:
        return None
    keys.sort(key=lambda k: (0 if "ISOMERIC" in k.upper() else 1))
    smi = str(props[keys[0]])
    if "." in smi:
        smi = max(smi.split("."), key=len)
    return smi or None

def smiles_from_pubchem(name):
    for v in dict.fromkeys([str(name).strip(), str(name).split(",")[0].strip(), norm_name(name)]):
        if v:
            smi = parse_smiles(fetch_json(PUBCHEM.format(urllib.parse.quote(v, safe=""))))
            if smi:
                return smi
    return None

def lookup(item):
    i, name = item
    try:
        s = smiles_from_pubchem(name)
        return i, (s or ""), ("ok" if s else "not_found")
    except Transient:
        return i, "", "unknown"

def run_pubchem(drugs, passes=3, workers=4):
    res = {}
    todo = list(zip(drugs["drug_id"].astype("int64"), drugs["drug_name"]))
    for p in range(passes):
        todo = [t for t in todo if t[0] not in res]
        if not todo:
            break
        print(f"pass {p+1}: asking PubChem for {len(todo)} drugs ...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            for k, (i, s, st) in enumerate(ex.map(lookup, todo), 1):
                if st != "unknown":
                    res[i] = (s, st)
                if k % 200 == 0:
                    print(f"   {k}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
        if p < passes - 1 and any(t[0] not in res for t in todo):
            time.sleep(20)
    return res

def morgan(smi):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    a = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), a)
    return a

def build(drugs, out="drug_fp_colab.npz"):
    res = run_pubchem(drugs)
    ids = drugs["drug_id"].astype("int64").to_numpy()
    smi = np.array([res.get(i, ("", "unknown"))[0] for i in ids])
    st = np.array([res.get(i, ("", "unknown"))[1] for i in ids])
    fp = np.zeros((len(ids), N_BITS), dtype=np.uint8)
    valid = np.zeros(len(ids), dtype=bool)
    for k, s in enumerate(smi):
        if s:
            a = morgan(s)
            if a is not None:
                fp[k] = a; valid[k] = True
            else:
                st[k] = "invalid_smiles"
    print("status:", pd.Series(st).value_counts().to_dict())
    print(f"valid fingerprints: {valid.sum()} of {len(ids)}   average bits set: {fp[valid].sum(1).mean():.1f}")
    np.savez_compressed(out, ids=ids, smiles=smi, status=st, fp=fp, valid=valid,
                        n_bits=N_BITS, radius=RADIUS)
    print("saved", out)
    return out


def build_from_work(work_dir="work"):
    """Read work/drugs_raw.csv, write work/drug_fp_colab.npz (skipped if it already exists)."""
    from pathlib import Path
    work = Path(work_dir)
    out = work / "drug_fp_colab.npz"
    if out.exists():
        print("already exists:", out)
        return out
    drugs = pd.read_csv(work / "drugs_raw.csv")
    build(drugs, str(out))
    return out
