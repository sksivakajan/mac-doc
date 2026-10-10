"""Helper code for the Component 3 notebooks (MedSafe-AI).

The notebooks show every step; this file holds the longer functions so the
notebooks stay short. Nothing here changes your original data files.
"""
import collections
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from severity import SEVERITY_LEVELS, SEVERITY_RANK, severity_of

SEED = 42
KEY_BASE = 10**9   # pair key = smaller_id * KEY_BASE + bigger_id


# ----------------------------------------------------------------------------
# 1. Finding files
# ----------------------------------------------------------------------------
def find_data_dir():
    """Folder that holds TWOSIDES.csv and OFFSIDES.csv (your 'data set' folder)."""
    here = Path.cwd()
    for c in [Path(r"D:\med\data set"), here, here.parent, here.parent.parent]:
        if (c / "TWOSIDES.csv").exists():
            return c
    raise FileNotFoundError("Cannot find TWOSIDES.csv. Set DATA_DIR by hand in the first cell.")


def find_file(data_dir, name):
    """Your download made 'TWOSIDES.csv' a FOLDER that contains 'TWOSIDES.csv'. Handle both."""
    p = Path(data_dir) / name
    if p.is_dir():
        p = p / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


# ----------------------------------------------------------------------------
# 2. Reading the big CSV files safely (in pieces)
# ----------------------------------------------------------------------------
def _column(header, prefix):
    """The real files have typos ('rxnorn' vs 'rxnorm'), so find columns by their start."""
    for c in header:
        if c.startswith(prefix):
            return c
    raise KeyError(prefix)


def read_clean_chunks(path, usecols, numeric_cols, stats, chunksize=1_000_000):
    """Yield pieces of a big CSV. Lines that are not real data (for example a header
    line repeated in the middle of the file) are dropped and counted in stats['bad_rows']."""
    stats.setdefault("rows_read", 0)
    stats.setdefault("bad_rows", 0)
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        n0 = len(ch)
        for c in numeric_cols:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = ch.dropna(subset=numeric_cols)
        stats["bad_rows"] += n0 - len(ch)
        stats["rows_read"] += len(ch)
        yield ch


def collect_names(need_drugs, need_effects, twosides_path, offsides_path=None, chunksize=1_000_000, log=print):
    """Find the names that belong to drug ids and side-effect ids.
    First OFFSIDES (small and fast); for ids it does not know, TWOSIDES is scanned until all are found."""
    need_d, need_e = set(int(x) for x in need_drugs), set(int(x) for x in need_effects)
    dn, en = {}, {}
    if offsides_path is not None:
        header = pd.read_csv(offsides_path, nrows=0).columns.tolist()
        c_id = _column(header, "drug_rxn")
        for ch in read_clean_chunks(offsides_path, [c_id, "drug_concept_name", "condition_meddra_id",
                                                    "condition_concept_name"], [c_id, "condition_meddra_id"], {}, chunksize):
            u = ch.drop_duplicates(c_id)
            dn.update(zip(u[c_id].astype("int64"), u["drug_concept_name"].astype(str)))
            u = ch.drop_duplicates("condition_meddra_id")
            en.update(zip(u["condition_meddra_id"].astype("int64"), u["condition_concept_name"].astype(str)))
    miss_d, miss_e = need_d - set(dn), need_e - set(en)
    if miss_d or miss_e:
        log(f"  names missing in OFFSIDES: {len(miss_d)} drugs, {len(miss_e)} side effects -> scanning TWOSIDES")
        header = pd.read_csv(twosides_path, nrows=0).columns.tolist()
        c_d1, c_d2 = _column(header, "drug_1_rxn"), _column(header, "drug_2_rxn")
        cols = [c_d1, "drug_1_concept_name", c_d2, "drug_2_concept_name", "condition_meddra_id", "condition_concept_name"]
        cat = {"drug_1_concept_name": "category", "drug_2_concept_name": "category", "condition_concept_name": "category"}
        for ch in pd.read_csv(twosides_path, usecols=cols, dtype=cat, chunksize=chunksize, low_memory=False):
            for idc, nmc in [(c_d1, "drug_1_concept_name"), (c_d2, "drug_2_concept_name")]:
                ids = pd.to_numeric(ch[idc], errors="coerce")
                ok = ids.notna() & ids.isin(miss_d)
                if ok.any():
                    u = ch[ok].assign(_id=ids[ok].astype("int64")).drop_duplicates("_id")
                    dn.update(zip(u["_id"], u[nmc].astype(str)))
            ids = pd.to_numeric(ch["condition_meddra_id"], errors="coerce")
            ok = ids.notna() & ids.isin(miss_e)
            if ok.any():
                u = ch[ok].assign(_id=ids[ok].astype("int64")).drop_duplicates("_id")
                en.update(zip(u["_id"], u["condition_concept_name"].astype(str)))
            miss_d, miss_e = need_d - set(dn), need_e - set(en)
            if not miss_d and not miss_e:
                break
    return {k: v for k, v in dn.items() if k in need_d}, {k: v for k, v in en.items() if k in need_e}


def load_twosides_signals(path, prr_min=2.0, a_min=3, offsides_path=None, chunksize=1_000_000, log=print):
    """Read TWOSIDES in pieces and keep only the strong signals (PRR >= prr_min and A >= a_min).
    Returns a dict of numpy arrays + name dictionaries + simple counts."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    c_d1 = _column(header, "drug_1_rxn")
    c_d2 = _column(header, "drug_2_rxn")
    cols = [c_d1, c_d2, "condition_meddra_id", "A", "PRR"]
    stats, t0 = {}, time.time()
    keep = {k: [] for k in ["d1", "d2", "eff", "A", "PRR"]}
    n_prr_ok = 0
    for ch in read_clean_chunks(path, cols, cols, stats, chunksize):
        n_prr_ok += int((ch["PRR"] >= prr_min).sum())
        s = ch[(ch["PRR"] >= prr_min) & (ch["A"] >= a_min)]
        keep["d1"].append(s[c_d1].to_numpy("int64"))
        keep["d2"].append(s[c_d2].to_numpy("int64"))
        keep["eff"].append(s["condition_meddra_id"].to_numpy("int64"))
        keep["A"].append(s["A"].to_numpy("int32"))
        keep["PRR"].append(s["PRR"].to_numpy("float32"))
        if stats["rows_read"] % 5_000_000 < chunksize:
            log(f"  rows read: {stats['rows_read']:,}   kept so far: {sum(len(x) for x in keep['d1']):,}"
                f"   ({time.time() - t0:.0f}s)")
    out = {k: np.concatenate(v) for k, v in keep.items()}
    lo = np.minimum(out["d1"], out["d2"])
    hi = np.maximum(out["d1"], out["d2"])
    out["lo"], out["hi"] = lo, hi
    log(f"  rows read: {stats['rows_read']:,}   kept: {len(lo):,}   ({time.time() - t0:.0f}s). Now looking up names ...")
    out["drug_names"], out["effect_names"] = collect_names(
        np.unique(np.concatenate([lo, hi])), np.unique(out["eff"]), path, offsides_path, chunksize, log)
    out["stats"] = dict(stats, rows_prr_ok=n_prr_ok, rows_kept=len(lo), prr_min=prr_min, a_min=a_min)
    return out


# ----------------------------------------------------------------------------
# 3. Pairs x side effects  ->  label matrix
# ----------------------------------------------------------------------------
def build_pairs_and_labels(sig, top_k=50, extra_high_min_pairs=500, extra_high_max=20):
    """One row per drug pair (order does not matter), one column per chosen side effect.
    Y[i, j] = 1 if TWOSIDES reports a strong signal for pair i and side effect j.
    Effects chosen: the top_k most common + a few rarer ones that look serious (High severity)."""
    lo, hi, eff = sig["lo"], sig["hi"], sig["eff"]
    ok = lo != hi
    lo, hi, eff = lo[ok], hi[ok], eff[ok]
    key = lo * KEY_BASE + hi
    pair_keys, inv = np.unique(key, return_inverse=True)
    n_pairs = len(pair_keys)

    pe = pd.DataFrame({"p": inv, "e": eff}).drop_duplicates()
    eff_counts = pe["e"].value_counts()                      # in how many pairs each effect appears
    names = sig["effect_names"]
    chosen = list(eff_counts.index[:top_k])
    extra = [e for e in eff_counts.index[top_k:]
             if eff_counts[e] >= extra_high_min_pairs and severity_of(names.get(e, "")) == "High"]
    chosen += extra[:extra_high_max]

    col_of = {e: j for j, e in enumerate(chosen)}
    pe = pe[pe["e"].isin(col_of)]
    Y = np.zeros((n_pairs, len(chosen)), dtype=np.uint8)
    Y[pe["p"].to_numpy(), pe["e"].map(col_of).to_numpy()] = 1

    pairs = pd.DataFrame({"drug_a_id": pair_keys // KEY_BASE, "drug_b_id": pair_keys % KEY_BASE})
    effects = pd.DataFrame({"effect_id": chosen,
                            "effect_name": [names.get(e, str(e)) for e in chosen],
                            "n_pairs": [int(eff_counts[e]) for e in chosen],
                            "severity": [severity_of(names.get(e, "")) or "Unknown" for e in chosen]})
    # pairs with no chosen effect are useless for training -> drop them
    has = Y.sum(axis=1) > 0
    return pairs[has].reset_index(drop=True), Y[has], effects


# ----------------------------------------------------------------------------
# 4. Drug names, openFDA labels (class / use / label side effects)
# ----------------------------------------------------------------------------
SALT_WORDS = {"hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "sulfate",
              "sulphate", "citrate", "mesylate", "maleate", "tartrate", "phosphate", "acetate",
              "besylate", "succinate", "fumarate", "bromide", "hydrobromide", "chloride",
              "nitrate", "oxalate", "lactate", "gluconate", "hydrate", "monohydrate", "dihydrate",
              "anhydrous", "usp", "disodium", "dipropionate", "propionate", "valerate", "bitartrate",
              "tosylate", "hemihydrate", "carbonate", "oxide", "hydroxide", "stearate", "palmitate",
              "decanoate", "cypionate", "enanthate", "medoxomil", "axetil", "pamoate", "mofetil"}


def norm_name(s):
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_variants(s):
    """The name itself, then the name with salt words removed ('sildenafil citrate' -> 'sildenafil')."""
    s = norm_name(s)
    out, words = [s], s.split()
    w = list(words)
    while len(w) > 1 and (w[-1] in SALT_WORDS or w[-1].rstrip("s") in SALT_WORDS):
        w = w[:-1]
        out.append(" ".join(w))
    w = [x for x in words if x not in SALT_WORDS and x.rstrip("s") not in SALT_WORDS]
    if w and " ".join(w) not in out:
        out.append(" ".join(w))
    return out


def iter_label_results(path, block=1 << 24):
    """Stream the label objects out of one big openFDA JSON file without loading the whole file
    (each file is ~600 MB). Needs only the standard library."""
    dec = json.JSONDecoder()
    start = re.compile(r'"results"\s*:\s*\[')
    skip = re.compile(r"[\s,]*")
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        while True:                                       # find where the results array starts
            more = f.read(block)
            if not more:
                return
            buf += more
            m = start.search(buf)
            if m:
                buf = buf[m.end():]
                break
            buf = buf[-64:]
        pos, eof = 0, False
        while True:
            pos = skip.match(buf, pos).end()
            if pos >= len(buf):
                more = f.read(block)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(block)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > block:                                # keep the buffer small
                buf, pos = buf[pos:], 0


def _first_text(label, field, limit):
    v = label.get(field)
    if isinstance(v, list) and v:
        v = v[0]
    if not isinstance(v, str):
        return ""
    v = re.sub(r"\s+", " ", v).strip()
    v = re.sub(r"^(\d+(\.\d+)*\s*)?(indications?\s*(and|&)\s*usage|adverse reactions)\s*:?\s*", "", v, flags=re.I)
    return v[:limit]


def scan_labels(files, wanted, log=print, max_labels=None):
    """Go through the openFDA label files and collect class / use / label side-effects text for
    the drugs in `wanted` (dict: normalised drug name -> drug id). Only single-ingredient labels are used."""
    acc = collections.defaultdict(lambda: {"classes": collections.Counter(), "uses": [], "adverse": [], "labels": 0})
    n = 0
    for fp in files:
        t0 = time.time()
        for lab in iter_label_results(fp):
            n += 1
            if max_labels and n > max_labels:
                break
            ofda = lab.get("openfda") or {}
            subs = ofda.get("substance_name") or ofda.get("generic_name") or []
            if len(subs) != 1:
                continue
            hit = next((wanted[v] for v in name_variants(subs[0]) if v in wanted), None)
            if hit is None:
                continue
            a = acc[hit]
            a["labels"] += 1
            for c in ofda.get("pharm_class_epc") or []:
                a["classes"][re.sub(r"\s*\[EPC\]\s*$", "", c)] += 1
            if len(a["uses"]) < 3:
                t = _first_text(lab, "indications_and_usage", 300)
                if t:
                    a["uses"].append(t)
            if len(a["adverse"]) < 2:
                t = _first_text(lab, "adverse_reactions", 1500)
                if t:
                    a["adverse"].append(t)
        log(f"  {Path(fp).name}: done ({time.time() - t0:.0f}s), labels seen so far {n:,}, drugs matched {len(acc):,}")
        if max_labels and n > max_labels:
            break
    rows = []
    for did, a in acc.items():
        rows.append({"drug_id": did,
                     "drug_class": a["classes"].most_common(1)[0][0] if a["classes"] else "",
                     "all_classes": "; ".join(c for c, _ in a["classes"].most_common(5)),
                     "main_use": a["uses"][0] if a["uses"] else "",
                     "label_adverse_reactions": a["adverse"][0] if a["adverse"] else "",
                     "n_labels": a["labels"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. SMILES from PubChem (needs internet) and fingerprints
# ----------------------------------------------------------------------------
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
UA = {"User-Agent": "medsafe-ai-student-project/1.0"}


class Transient(Exception):
    """PubChem was busy or the internet failed. This is NOT the same as 'drug not found'."""


_last_call = [0.0]


def _polite_wait(gap=0.25):
    """At most 4 requests per second in total (PubChem allows 5)."""
    wait = gap - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def fetch_json(url, retries=5, pause=3.0):
    """Parsed JSON; None if the thing really does not exist (404 / bad name);
    raises Transient if the server is busy or the internet fails (so we never confuse the two)."""
    for attempt in range(retries):
        _polite_wait()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(pause * (2 ** attempt))
                continue
            raise Transient(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(pause * (2 ** attempt))
    raise Transient("server busy or no internet")


def parse_pubchem_smiles(data):
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
    """SMILES or None (= PubChem really does not know the name). Raises Transient on temporary problems."""
    for v in dict.fromkeys([str(name).strip(), str(name).split(",")[0].strip(), norm_name(name)]):
        if not v:
            continue
        smi = parse_pubchem_smiles(fetch_json(PUBCHEM.format(urllib.parse.quote(v, safe=""))))
        if smi:
            return smi
    return None


def get_smiles(drug_table, cache_path, fetch=smiles_from_pubchem, max_in_a_row=25, log=print):
    """Adds columns 'smiles' and 'smiles_status' ('ok', 'not_found' or 'unknown') to drug_table
    (needs columns drug_id, drug_name). Progress is remembered in cache_path.
    Only real answers are remembered. Temporary failures stay 'unknown' and are tried again next time.
    (An old cache without a 'status' column: empty entries are treated as unknown and tried again.)"""
    cache = {}
    if Path(cache_path).exists():
        c = pd.read_csv(cache_path, keep_default_na=False)
        for i, smi, st in zip(c["drug_id"].astype("int64"), c["smiles"].astype(str),
                              c["status"] if "status" in c else [None] * len(c)):
            if smi:
                cache[i] = (smi, "ok")
            elif st == "not_found":
                cache[i] = ("", "not_found")

    def save():
        pd.DataFrame({"drug_id": list(cache), "smiles": [v[0] for v in cache.values()],
                      "status": [v[1] for v in cache.values()]}).to_csv(cache_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    todo = [(i, n) for i, n in zip(drug_table["drug_id"], drug_table["drug_name"]) if i not in cache]
    log(f"  SMILES: {len(cache)} already known, {len(todo)} to ask PubChem (about {len(todo) * 0.35 / 60:.0f} minutes)")
    in_a_row = 0
    for k, (i, n) in enumerate(todo, 1):
        try:
            smi = fetch(n)
            cache[i] = (smi or "", "ok" if smi else "not_found")
            in_a_row = 0
        except Transient as e:
            in_a_row += 1
            if in_a_row >= max_in_a_row:
                log(f"  !! PubChem keeps failing ({e}). Stopping now. Run this cell again later; progress is saved.")
                break
            time.sleep(5)
        if k % 50 == 0:
            save()
            log(f"    {k}/{len(todo)}")
    save()
    out = drug_table.copy()
    out["smiles"] = out["drug_id"].map({i: v[0] for i, v in cache.items()}).fillna("")
    out["smiles_status"] = out["drug_id"].map({i: v[1] for i, v in cache.items()}).fillna("unknown")
    return out


_rdkit_error = [""]


def have_rdkit():
    """True if RDKit can be imported. Checked every time, so it also works right after 'pip install rdkit'
    without restarting the kernel."""
    import importlib
    importlib.invalidate_caches()
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception as e:
        _rdkit_error[0] = f"{type(e).__name__}: {e}"
        return False


def rdkit_error():
    """Why the last RDKit import failed."""
    return _rdkit_error[0]


def morgan_fp(smiles, radius=2, n_bits=1024):
    """One drug -> 0/1 vector. Needs RDKit. Returns None for an invalid SMILES."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception:
        raise RuntimeError("RDKit is not installed. Run:  pip install rdkit")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    return arr


def hashed_smiles_fp(smiles, n_bits=1024):
    """CRUDE stand-in (hashed 3-letter pieces of the SMILES text). Only for testing the code
    when RDKit is missing. Do not use for real results."""
    fp = np.zeros(n_bits, dtype=np.uint8)
    s = str(smiles)
    for i in range(len(s) - 2):
        fp[zlib.crc32(s[i:i + 3].encode()) % n_bits] = 1
    return fp


# ----------------------------------------------------------------------------
# 6. Pair features (the same for A+B and B+A)
# ----------------------------------------------------------------------------
def pair_fp_matrix(idx_a, idx_b, drug_fp):
    """[A OR B , A AND B] of the two drug fingerprints. drug_fp: (n_drugs, n_bits)."""
    a, b = drug_fp[idx_a], drug_fp[idx_b]
    return np.hstack([a | b, a & b]).astype(np.uint8)


def class_matrix(drug_classes, top_m=100):
    """Multi-hot matrix (n_drugs, M) of the most common drug classes + the class names."""
    cnt = collections.Counter(c for cs in drug_classes for c in cs)
    names = [c for c, _ in cnt.most_common(top_m)]
    pos = {c: j for j, c in enumerate(names)}
    M = np.zeros((len(drug_classes), len(names)), dtype=np.uint8)
    for i, cs in enumerate(drug_classes):
        for c in cs:
            if c in pos:
                M[i, pos[c]] = 1
    return M, names


class DrugHistory:
    """'How often did this drug cause side effect j in the TRAINING pairs?'  (idea of your friend's best notebook)
    - fit() uses TRAINING pairs only.
    - For training rows we remove the pair's own label (leave-one-out), otherwise the model could
      read the answer from its own feature. For validation/test rows nothing is removed (they were never used).
    - shrink pulls rare drugs towards the overall average."""

    def __init__(self, n_drugs, shrink=5.0):
        self.n_drugs, self.shrink = n_drugs, shrink

    def fit(self, idx_a, idx_b, Y):
        K = Y.shape[1]
        self.sums = np.zeros((self.n_drugs, K), dtype=np.float64)
        self.cnt = np.zeros(self.n_drugs, dtype=np.float64)
        for idx in (idx_a, idx_b):
            np.add.at(self.sums, idx, Y)
            np.add.at(self.cnt, idx, 1)
        self.prior = Y.mean(axis=0)
        return self

    def transform(self, idx_a, idx_b, Y_own=None):
        def one(idx):
            s, c = self.sums[idx].copy(), self.cnt[idx].copy()
            if Y_own is not None:                               # leave-one-out for training rows
                s -= Y_own
                c -= 1
            return (s + self.shrink * self.prior) / (c[:, None] + self.shrink), c
        ha, ca = one(idx_a)
        hb, cb = one(idx_b)
        feats = [np.minimum(ha, hb), np.maximum(ha, hb),
                 np.log1p(np.minimum(ca, cb))[:, None], np.log1p(np.maximum(ca, cb))[:, None]]
        return np.hstack(feats).astype(np.float32)


# ----------------------------------------------------------------------------
# 7. Splitting 70 / 20 / 10 and checking the split
# ----------------------------------------------------------------------------
def split_random(n, fracs=(0.7, 0.2, 0.1), seed=SEED):
    """Random split of PAIRS. Every pair (with all its side effects) lands in exactly one part."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tr, n_va = int(round(n * fracs[0])), int(round(n * fracs[1]))
    return order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]


def split_unseen_drug(idx_a, idx_b, fracs=(0.7, 0.2, 0.1), seed=SEED):
    """Harder split: some DRUGS are kept out of training completely.
    test pairs  = pairs with at least one 'test drug'; val pairs = pairs with at least one 'val drug'
    (and no test drug); train = pairs where both drugs are normal. Drugs are added at random until the
    shares are about 10% (test) and 20% (val) of all pairs."""
    n = len(idx_a)
    rng = np.random.default_rng(seed)
    drugs = rng.permutation(np.unique(np.concatenate([idx_a, idx_b])))
    is_test = np.zeros(drugs.max() + 1, bool)
    is_val = np.zeros(drugs.max() + 1, bool)
    it = iter(drugs)
    while (is_test[idx_a] | is_test[idx_b]).mean() < fracs[2]:
        is_test[next(it)] = True
    test_mask = is_test[idx_a] | is_test[idx_b]
    while ((is_val[idx_a] | is_val[idx_b]) & ~test_mask).mean() < fracs[1]:
        d = next(it)
        is_val[d] = True
    val_mask = (is_val[idx_a] | is_val[idx_b]) & ~test_mask
    train_mask = ~test_mask & ~val_mask
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def check_split(idx_a, idx_b, Y, parts, effects):
    """Data checks after splitting. Returns a dict of tables/numbers; the notebook prints them."""
    names = ["train", "val", "test"]
    key = np.minimum(idx_a, idx_b).astype("int64") * KEY_BASE + np.maximum(idx_a, idx_b)
    rep = {"sizes": {}, "pair_overlap": {}, "drug_overlap": [], "labels": [], "zero_pos_effects": {}}
    for nm, ix in zip(names, parts):
        rep["sizes"][nm] = len(ix)
    sets = [set(key[ix].tolist()) for ix in parts]
    rep["duplicate_pairs_total"] = int(len(key) - len(set(key.tolist())))
    for i in range(3):
        for j in range(i + 1, 3):
            rep["pair_overlap"][f"{names[i]} & {names[j]}"] = len(sets[i] & sets[j])
    train_drugs = set(idx_a[parts[0]].tolist()) | set(idx_b[parts[0]].tolist())
    for nm, ix in zip(names, parts):
        a_new = ~np.isin(idx_a[ix], list(train_drugs))
        b_new = ~np.isin(idx_b[ix], list(train_drugs))
        rep["drug_overlap"].append({"part": nm, "pairs": len(ix),
                                    "pairs_with_an_unseen_drug_%": round(100 * float((a_new | b_new).mean()), 2) if len(ix) else 0,
                                    "pairs_with_two_unseen_drugs_%": round(100 * float((a_new & b_new).mean()), 2) if len(ix) else 0})
        y = Y[ix]
        rep["labels"].append({"part": nm, "pairs": len(ix), "avg_effects_per_pair": round(float(y.sum(1).mean()), 2),
                              "positive_density_%": round(100 * float(y.mean()), 2)})
        rep["zero_pos_effects"][nm] = [effects["effect_name"].iloc[j] for j in np.where(y.sum(0) == 0)[0]]
    return rep


# ----------------------------------------------------------------------------
# 8. Metrics
# ----------------------------------------------------------------------------
def precision_recall_at_k(Y, S, k):
    """Mean Precision@k and Recall@k over pairs (recall only over pairs that have at least 1 true effect)."""
    k = min(k, S.shape[1])
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(Y, top, axis=1).sum(1)
    npos = Y.sum(1)
    ok = npos > 0
    return float((hits / k).mean()), float((hits[ok] / npos[ok]).mean())


def best_threshold(Y, S, grid=None):
    """Score cut-off that gives the best micro-F1 on the VALIDATION set."""
    Yb = Y.astype(bool)
    grid = np.unique(np.quantile(S, np.linspace(0.5, 0.995, 40))) if grid is None else grid
    best, best_f = float(grid[0]), -1.0
    n_pos = Yb.sum()
    for t in grid:
        P = S >= t
        tp = np.count_nonzero(P & Yb)
        f = 2 * tp / max(P.sum() + n_pos, 1)
        if f > best_f:
            best, best_f = float(t), f
    return best


def evaluate(Y, S, thr, high_mask=None):
    """All metrics for one model on one part. Y: 0/1 truth, S: scores."""
    from sklearn.metrics import average_precision_score, f1_score
    Yb = Y.astype(bool)
    P = S >= thr
    has_pos = Y.sum(0) > 0
    p3, _ = precision_recall_at_k(Y, S, 3)
    p5, _ = precision_recall_at_k(Y, S, 5)
    p10, r10 = precision_recall_at_k(Y, S, 10)
    out = {"micro_AUPRC": average_precision_score(Y.ravel(), S.ravel()),
           "macro_AUPRC": float(np.mean([average_precision_score(Y[:, j], S[:, j]) for j in np.where(has_pos)[0]])),
           "P@3": p3, "P@5": p5, "P@10": p10, "R@10": r10,
           "micro_F1": f1_score(Y.ravel(), P.ravel(), zero_division=0),
           "macro_F1": f1_score(Y[:, has_pos], P[:, has_pos], average="macro", zero_division=0),
           "hamming_loss": float((Yb != P).mean())}
    if high_mask is not None and high_mask.any():
        tp = (Yb[:, high_mask] & P[:, high_mask]).sum()
        out["recall_High_effects"] = float(tp / max(Yb[:, high_mask].sum(), 1))
    return out


def pair_severity(Yb, eff_rank):
    """Worst severity among a pair's effects: 0 Low, 1 Medium, 2 High, -1 = none/unknown."""
    r = np.where(Yb.astype(bool), eff_rank[None, :], -1)
    return r.max(axis=1)


# ----------------------------------------------------------------------------
# 9. Models: baseline + turning any model's output into scores
# ----------------------------------------------------------------------------
class PrevalenceBaseline:
    """Dummy model: every pair gets the same score per side effect (how common it is in TRAINING)."""

    def fit(self, X, Y):
        self.p_ = Y.mean(axis=0)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (X.shape[0], 1))


def predict_scores(model, X):
    """(n_pairs, n_effects) scores for any model: probabilities when the model has them,
    otherwise a sigmoid of the decision value (SVM)."""
    if hasattr(model, "predict_proba"):
        out = model.predict_proba(X)
        if isinstance(out, list):                  # RandomForest multi-label: one array per effect
            cols = []
            for j, p in enumerate(out):
                if p.shape[1] == 2:
                    cols.append(p[:, 1])
                else:                              # effect had a single class in training
                    cols.append(np.full(p.shape[0], float(model.classes_[j][0] == 1)))
            return np.column_stack(cols)
        return np.asarray(out)
    d = np.asarray(model.decision_function(X))
    return 1.0 / (1.0 + np.exp(-d))
