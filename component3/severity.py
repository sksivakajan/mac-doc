"""Severity labelling: side-effect term -> Low / Medium / High.

IMPORTANT: TWOSIDES/OFFSIDES do NOT contain severity. The labels below are a
STARTING POINT (hand-made exact terms + keyword rules). Before you report any
severity result, agree the real source with your supervisor (e.g. MedDRA
seriousness grouping, or an expert-reviewed list) and replace/extend this file.
"""
import re

SEVERITY_LEVELS = ["Low", "Medium", "High"]
SEVERITY_RANK = {"Low": 0, "Medium": 1, "High": 2}

# 1) Exact terms (lower-case). Used by the sample data.
EXACT = {
    # High
    "gastrointestinal haemorrhage": "High", "serotonin syndrome": "High",
    "respiratory depression": "High", "hyperkalaemia": "High",
    "rhabdomyolysis": "High", "hypoglycaemia": "High",
    "qt interval prolonged": "High", "acute kidney injury": "High",
    # Medium
    "hypotension": "Medium", "dizziness": "Medium", "somnolence": "Medium",
    "bradycardia": "Medium", "hepatic enzyme increased": "Medium",
    "tachycardia": "Medium", "hypokalaemia": "Medium", "myalgia": "Medium",
    "syncope": "Medium",
    # Low
    "nausea": "Low", "headache": "Low", "diarrhoea": "Low", "dry mouth": "Low",
    "constipation": "Low", "rash": "Low", "fatigue": "Low", "vomiting": "Low",
    "abdominal pain": "Low", "insomnia": "Low", "cough": "Low",
}

# 2) Keyword rules for the real data (1000+ terms). Checked High -> Medium -> Low,
#    first match wins. These are rough heuristics, NOT clinical truth.
_HIGH = [r"h?a?emorrhag", r"bleed", r"failure", r"arrest", r"anaphyla", r"torsade",
         r"\bqt\b", r"rhabdomyolysis", r"hyperkal", r"hypoglyc", r"infarct", r"stroke",
         r"embol", r"thrombo", r"sepsis", r"\bcoma\b", r"seizure", r"convulsion",
         r"death", r"fibrillation", r"angioedema", r"necrosis", r"perforation",
         r"serotonin syndrome", r"respiratory depression", r"kidney injury",
         r"stevens", r"agranulocytosis", r"suicid", r"overdose"]
_MEDIUM = [r"hypotens", r"dizz", r"somnol", r"sedat", r"bradycard", r"tachycard",
           r"hepat", r"liver", r"transaminase", r"enzyme increased", r"hypokal",
           r"myalg", r"syncope", r"an?emia", r"infection", r"pneumonia", r"dehydrat",
           r"confusion", r"tremor", r"palpitation", r"o?edema", r"dyspn",
           r"blood pressure", r"hypertens", r"depression", r"anxiety", r"chest pain",
           r"vision", r"renal"]
_LOW = [r"nausea", r"headache", r"diarrh", r"dry mouth", r"constipation", r"rash",
        r"fatigue", r"vomit", r"abdominal pain", r"insomnia", r"cough", r"prurit",
        r"malaise", r"asthenia", r"flatulence", r"dyspepsia", r"\bpain\b",
        r"decreased appetite", r"hot flush", r"sweating"]
_RULES = [("High", [re.compile(p) for p in _HIGH]),
          ("Medium", [re.compile(p) for p in _MEDIUM]),
          ("Low", [re.compile(p) for p in _LOW])]


def severity_of(term):
    """Return 'Low' / 'Medium' / 'High', or None if the term is not mapped."""
    t = str(term).lower().strip()
    if t in EXACT:
        return EXACT[t]
    for level, patterns in _RULES:
        if any(p.search(t) for p in patterns):
            return level
    return None


def severity_ranks(terms):
    """Array of 0/1/2 for each term, or -1 if unmapped."""
    out = []
    for t in terms:
        s = severity_of(t)
        out.append(SEVERITY_RANK[s] if s else -1)
    return out


def coverage(terms):
    """Share of terms that got a severity label (report this in your thesis)."""
    ranks = severity_ranks(terms)
    return sum(r >= 0 for r in ranks) / max(len(ranks), 1)
