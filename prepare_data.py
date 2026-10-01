"""
prepare_data.py — build a privacy-safe applicants.csv for the triage app.

WHAT THIS DOES
  1. Loads an open dataset (default: UCI Adult / Census Income — fully open, no login).
  2. Maps raw columns to the app's welfare schema.
  3. MASKS PII: drops direct identifiers, stores only a salted hash as a surrogate,
     segregates protected attributes, and builds a de-identified narrative that
     contains no name, no ID, and no protected attribute — this is the only text
     the LLM ever sees.
  4. Samples N rows and writes applicants.csv + prints a masking report.

USAGE
  python prepare_data.py --source adult  --n 500      # downloads UCI Adult
  python prepare_data.py --source kaggle --path train.csv --n 500   # IDB Costa Rican set
  python prepare_data.py --source sample --n 500      # offline simulated data (no network)

Open data sources
  - UCI Adult / Census Income (used here): the standard algorithmic-fairness dataset.
    https://archive.ics.uci.edu/dataset/2/adult
  - IDB "Costa Rican Household Poverty Level Prediction" on Kaggle — the most
    on-topic welfare-targeting set (household income, overcrowding, dependency,
    disability, poverty label). Download its train.csv, then use --source kaggle.
    https://www.kaggle.com/c/costa-rican-household-poverty-prediction
"""
import argparse, hashlib, io, sys
import numpy as np
import pandas as pd

SALT = "welfare-cycle-2026Q3"   # in production, keep the salt in a secret store
PROTECTED = ["sex", "race", "native_region"]   # monitored for fairness, NEVER scored or sent to the LLM

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
              "hours_per_week", "native_country", "income"]

REGION_MAP = {  # coarse buckets so we never store fine-grained origin
    "United-States": "Domestic", "Outlying-US(Guam-USVI-etc)": "Domestic",
}


def mask_id(raw_id):
    """Any real identifier is stored ONLY as a salted, truncated hash."""
    return "ID#" + hashlib.sha256(f"{SALT}:{raw_id}".encode()).hexdigest()[:12]


def to_region(country):
    c = str(country).strip()
    if c in REGION_MAP:
        return REGION_MAP[c]
    if c in ("?", "nan", ""):
        return "Undisclosed"
    return "International"


def build_narrative(r):
    """De-identified free text — the ONLY applicant text the LLM receives.
    Contains no name, no ID, and no protected attribute (sex/race/origin)."""
    parts = [f"Applicant reports annual income in the "
             f"{'lower band (at or below the means-test threshold)' if r['income_bracket']=='<=50K' else 'higher band (above the means-test threshold)'}."]
    hrs = int(r["hours_per_week"])
    if hrs < 20:
        parts.append(f"Works about {hrs} hours per week — well below full time.")
    elif hrs < 35:
        parts.append(f"Works about {hrs} hours per week — part time.")
    else:
        parts.append(f"Works about {hrs} hours per week.")
    wc = r["workclass"]
    if wc in ("Without-pay", "Never-worked", "?"):
        parts.append("Employment is precarious or unpaid.")
    elif "Self-emp" in str(wc):
        parts.append("Self-employed, with variable earnings.")
    parts.append(f"Completed roughly {int(r['education_num'])} years of schooling.")
    if int(r["capital_gain"]) == 0:
        parts.append("Reports no investment or capital income.")
    if int(r["capital_loss"]) > 0:
        parts.append("Reports recent capital losses.")
    return " ".join(parts)


def map_rows(raw):
    out = pd.DataFrame()
    out["id"] = [f"APP-{1000+i}" for i in range(len(raw))]
    out["pii_hash"] = [mask_id(x) for x in raw.get("fnlwgt", pd.Series(range(len(raw))))]  # stand-in for a real record id
    out["age"] = raw["age"].astype(int)
    out["education_num"] = raw["education_num"].astype(int)
    out["hours_per_week"] = raw["hours_per_week"].astype(int)
    out["income_bracket"] = raw["income"].str.contains("<=50K").map({True: "<=50K", False: ">50K"})
    out["workclass"] = raw["workclass"].str.strip()
    out["occupation"] = raw["occupation"].str.strip()
    out["marital_status"] = raw["marital_status"].str.strip()
    out["capital_gain"] = raw["capital_gain"].astype(int)
    out["capital_loss"] = raw["capital_loss"].astype(int)
    out["resident"] = raw["native_country"].apply(lambda c: to_region(c) != "International") | True  # treat all as resident for demo
    # protected attributes — segregated, monitored only
    out["sex"] = raw["sex"].str.strip()
    out["race"] = raw["race"].str.strip()
    out["native_region"] = raw["native_country"].apply(to_region)
    out["narrative"] = out.apply(build_narrative, axis=1)
    return out


def load_adult():
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    print(f"Downloading UCI Adult from {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = pd.read_csv(io.StringIO(resp.read().decode()), header=None, names=ADULT_COLS,
                          skipinitialspace=True, na_values="?")
    return raw


def load_kaggle(path):
    """Map the IDB Costa Rican Household Poverty set (train.csv) to Adult-like raw columns.
    Best-effort; verify column names against your downloaded file."""
    df = pd.read_csv(path)
    raw = pd.DataFrame({
        "age": df.get("age", 40),
        "workclass": "Private",
        "fnlwgt": df.index,
        "education": "NA",
        "education_num": df.get("escolari", df.get("meaneduc", 6)).fillna(6),
        "marital_status": "NA",
        "occupation": "NA",
        "relationship": "NA",
        "race": "Undisclosed",                 # set has no race; kept as placeholder protected field
        "sex": np.where(df.get("male", 1) == 1, "Male", "Female"),
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        # poverty target 1=extreme..4=non-vulnerable -> income band proxy
        "income": np.where(df.get("Target", 4) <= 2, "<=50K", ">50K"),
    })
    return raw


def simulate_adult(n, seed=7):
    """Offline Adult-like generator (no network) — identical schema to the real file."""
    rng = np.random.default_rng(seed)
    edu = rng.integers(1, 17, n)
    hrs = np.clip(rng.normal(38, 14, n).round(), 1, 80).astype(int)
    p_high = 1 / (1 + np.exp(-(0.25 * (edu - 10) + 0.04 * (hrs - 40))))
    income = np.where(rng.random(n) < p_high * 0.55, ">50K", "<=50K")
    cg = np.where(rng.random(n) < 0.12, rng.integers(1000, 20000, n), 0)
    return pd.DataFrame({
        "age": rng.integers(17, 90, n),
        "workclass": rng.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov",
                                 "Self-emp-inc", "Without-pay", "Never-worked", "?"], n,
                                p=[.62, .11, .07, .05, .04, .03, .03, .05]),
        "fnlwgt": rng.integers(20000, 400000, n),
        "education": "NA",
        "education_num": edu,
        "marital_status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced",
                                       "Separated", "Widowed"], n, p=[.46, .33, .13, .04, .04]),
        "occupation": rng.choice(["Service", "Craft-repair", "Sales", "Exec-managerial",
                                  "Machine-op", "Other", "?"], n),
        "relationship": "NA",
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                           n, p=[.68, .12, .09, .05, .06]),
        "sex": rng.choice(["Male", "Female"], n, p=[.52, .48]),
        "capital_gain": cg,
        "capital_loss": np.where(rng.random(n) < 0.05, rng.integers(500, 3000, n), 0),
        "hours_per_week": hrs,
        "native_country": rng.choice(["United-States", "Mexico", "Philippines", "India", "?"],
                                     n, p=[.82, .07, .03, .03, .05]),
        "income": income,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["adult", "kaggle", "sample"], default="adult")
    ap.add_argument("--path", default=None, help="path to Kaggle train.csv when --source kaggle")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="applicants.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if a.source == "adult":
        raw = load_adult()
    elif a.source == "kaggle":
        if not a.path:
            sys.exit("--path to the Kaggle train.csv is required for --source kaggle")
        raw = load_kaggle(a.path)
    else:
        raw = simulate_adult(a.n * 3, a.seed)

    raw = raw.dropna(subset=["age", "hours_per_week", "education_num"]).reset_index(drop=True)
    raw = raw.sample(min(a.n, len(raw)), random_state=a.seed).reset_index(drop=True)
    df = map_rows(raw)

    # ---- masking verification: narrative must not leak PII / protected terms ----
    banned = ["Male", "Female", "White", "Black", "Asian", "Indian", "Eskimo",
              "United-States", "Mexico", "Philippines"]
    leak = df["narrative"].str.contains("|".join(banned), case=False, na=False)
    assert not leak.any(), "Narrative leaked a protected/PII term — fix build_narrative()."

    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df)} rows -> {a.out}")
    print("Masking report:")
    print("  - Direct identifiers: none stored (only surrogate APP-id + salted hash column 'pii_hash').")
    print(f"  - Protected attributes segregated (monitored only, never scored/sent to LLM): {PROTECTED}")
    print("  - Narrative verified free of names, IDs and protected terms (the only text the LLM sees).")
    print(f"  - Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
