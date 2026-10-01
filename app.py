"""
Social Welfare Grant — AI-Assisted Application Triage
Single-file Streamlit prototype. Open data + a FREE open-weight LLM (Ollama).

Run locally (with the free LLM):
  1) pip install -r requirements.txt
  2) Install Ollama (https://ollama.com) and pull a small open model:
        ollama pull llama3.2        # or qwen2.5:3b, phi3.5, mistral
     Ollama then serves an OpenAI-compatible API at http://localhost:11434/v1
  3) python prepare_data.py --source adult --n 500     # builds applicants.csv
  4) streamlit run app.py

Deploy a shareable URL (Streamlit Community Cloud): commit app.py, requirements.txt
and applicants.csv to a public GitHub repo, then deploy at https://share.streamlit.io.
Note: a LOCAL Ollama is not reachable from Streamlit Cloud. For AI features on a hosted
URL, point the app at any free OpenAI-compatible open-weight endpoint via Secrets:
      LLM_BASE_URL = "https://api.groq.com/openai/v1"   # free tier, serves Llama etc.
      LLM_MODEL    = "llama-3.1-8b-instant"
      LLM_API_KEY  = "gsk_..."
The triage workflow (ranking, fairness, audit, export) runs fully WITHOUT any LLM.
"""
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

POVERTY_LABEL = "means-test threshold"
TOTAL, SURFACE, TARGET = 500, 100, 50
PROTECTED = ["sex", "race", "native_region"]   # monitored for fairness; NEVER scored or sent to the LLM

st.set_page_config(page_title="Welfare Grant Triage", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- config / LLM
def cfg(key, default):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


LLM_BASE_URL = cfg("LLM_BASE_URL", "http://localhost:11434/v1")   # Ollama default
LLM_MODEL = cfg("LLM_MODEL", "llama3.2")
LLM_API_KEY = cfg("LLM_API_KEY", "ollama")


def _headers():
    h = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        h["Authorization"] = f"Bearer {LLM_API_KEY}"
    return h


@st.cache_data(ttl=20, show_spinner=False)
def llm_available():
    try:
        r = requests.get(LLM_BASE_URL.rstrip("/") + "/models", headers=_headers(), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_llm(system, messages, max_tokens=500, temperature=0):
    payload = {"model": LLM_MODEL, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}] + messages}
    r = requests.post(LLM_BASE_URL.rstrip("/") + "/chat/completions", headers=_headers(), json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- data loading
@st.cache_data
def simulate(n=TOTAL, seed=7):
    """Offline fallback identical in schema to prepare_data.py output."""
    rng = np.random.default_rng(seed)
    edu = rng.integers(1, 17, n)
    hrs = np.clip(rng.normal(38, 14, n).round(), 1, 80).astype(int)
    p_high = 1 / (1 + np.exp(-(0.25 * (edu - 10) + 0.04 * (hrs - 40))))
    income = np.where(rng.random(n) < p_high * 0.55, ">50K", "<=50K")
    cg = np.where(rng.random(n) < 0.12, rng.integers(1000, 20000, n), 0)
    wc = rng.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc",
                     "Without-pay", "Never-worked", "?"], n, p=[.62, .11, .07, .05, .04, .03, .03, .05])
    df = pd.DataFrame({
        "id": [f"APP-{1000+i}" for i in range(n)],
        "pii_hash": [f"ID#{abs(hash((seed, i))) % 10**12:012d}" for i in range(n)],
        "age": rng.integers(17, 90, n), "education_num": edu, "hours_per_week": hrs,
        "income_bracket": income, "workclass": wc,
        "occupation": rng.choice(["Service", "Craft-repair", "Sales", "Exec-managerial", "Machine-op", "Other", "?"], n),
        "marital_status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"], n),
        "capital_gain": cg, "capital_loss": np.where(rng.random(n) < 0.05, rng.integers(500, 3000, n), 0),
        "resident": True,
        "sex": rng.choice(["Male", "Female"], n, p=[.52, .48]),
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], n, p=[.68, .12, .09, .05, .06]),
        "native_region": rng.choice(["Domestic", "International", "Undisclosed"], n, p=[.82, .13, .05]),
    })
    def narr(r):
        p = [f"Applicant reports annual income in the {'lower band (at or below the ' + POVERTY_LABEL + ')' if r['income_bracket']=='<=50K' else 'higher band (above the ' + POVERTY_LABEL + ')'}."]
        h = int(r["hours_per_week"])
        p.append(f"Works about {h} hours per week{' — well below full time' if h < 20 else ' — part time' if h < 35 else ''}.")
        if r["workclass"] in ("Without-pay", "Never-worked", "?"): p.append("Employment is precarious or unpaid.")
        p.append(f"Completed roughly {int(r['education_num'])} years of schooling.")
        if int(r["capital_gain"]) == 0: p.append("Reports no investment or capital income.")
        return " ".join(p)
    df["narrative"] = df.apply(narr, axis=1)
    return df


@st.cache_data
def load_data():
    for path in ("applicants.csv", "data/applicants.csv"):
        if os.path.exists(path):
            return pd.read_csv(path), path
    return simulate(), "simulated (run prepare_data.py for real open data)"


# ------------------------------------------------------- Need Index (transparent)
WEIGHTS = {"Low income": 34, "Underemployment": 20, "Limited education": 14,
           "Precarious work": 14, "Age vulnerability": 10, "No capital buffer": 8}
PRECARIOUS = {"Without-pay": 1.0, "Never-worked": 1.0, "?": 0.9, "Self-emp-not-inc": 0.5,
              "Private": 0.4, "Self-emp-inc": 0.15, "Local-gov": 0.2, "State-gov": 0.2, "Federal-gov": 0.2}


def factor_frame(df):
    low_income = np.where(df["income_bracket"] == "<=50K", 0.9, 0.15) + np.where(df["capital_gain"] == 0, 0.1, 0)
    age_vuln = np.maximum(np.clip((df["age"] - 60) / 20, 0, 1), np.clip((24 - df["age"]) / 8, 0, 1))
    return pd.DataFrame({
        "Low income": np.clip(low_income, 0, 1),
        "Underemployment": np.clip((40 - df["hours_per_week"]) / 40, 0, 1),
        "Limited education": np.clip((12 - df["education_num"]) / 12, 0, 1),
        "Precarious work": df["workclass"].map(PRECARIOUS).fillna(0.3),
        "Age vulnerability": age_vuln,
        "No capital buffer": np.where(df["capital_gain"] == 0, 1.0, 0.0),
    }, index=df.index)


def compute_scores(df, weights):
    f = factor_frame(df)
    tot = sum(weights.values())
    contrib = f.mul(pd.Series(weights)) / tot * 100
    out = df.copy()
    out["score"] = contrib.sum(axis=1).round().astype(int)
    for c in weights:
        out[f"c::{c}"] = contrib[c].round(1)
    out["eligible"] = df["resident"].astype(bool) & ~((df["income_bracket"] == ">50K") & (df["capital_gain"] > 7000))
    return out


def plain_reasons(r):
    out = []
    if r["income_bracket"] == "<=50K": out.append("Income at/below the means-test threshold")
    if r["hours_per_week"] < 35: out.append(f"Works only {int(r['hours_per_week'])} hrs/week")
    if r["education_num"] < 10: out.append(f"{int(r['education_num'])} years of schooling")
    if r["workclass"] in ("Without-pay", "Never-worked", "?"): out.append("Precarious/unpaid work")
    if r["age"] >= 60 or r["age"] <= 22: out.append(f"Age-related vulnerability ({int(r['age'])})")
    if r["capital_gain"] == 0: out.append("No capital/asset buffer")
    return out


def llm_safe_facts(r):
    """Build model-facing facts with NO PII and NO protected attributes."""
    return (f"income band {r['income_bracket']}; works {int(r['hours_per_week'])} hrs/week; "
            f"{int(r['education_num'])} years schooling; workclass {r['workclass']}; "
            f"age band {'elderly' if r['age']>=60 else 'youth' if r['age']<=24 else 'working-age'}; "
            f"capital income {'none' if r['capital_gain']==0 else 'some'}.")


# ---------------------------------------------------------------- state
for k, v in {"decisions": {}, "audit": [], "chat": []}.items():
    st.session_state.setdefault(k, v)


def log(aid, action, rank, score, note=""):
    st.session_state.audit.insert(0, {"Time": pd.Timestamp.now().strftime("%H:%M:%S"), "Applicant": aid,
                                      "AI rank": rank, "AI score": score, "Action": action, "Officer": "R. Menon", "Note": note})


def selected_count():
    return sum(1 for d in st.session_state.decisions.values() if d.get("status") == "selected")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scoring policy")
    st.caption("Weights are set by policy, not learned. Changing them re-ranks the queue; the logic stays auditable.")
    w = {k: st.slider(k, 0, 40, v) for k, v in WEIGHTS.items()}
    st.divider()
    st.markdown("**🔒 Privacy by design**")
    st.caption("Applicants appear as surrogate IDs only. Direct identifiers are hashed at storage. "
               "Protected attributes (sex, race, origin) are **monitored for fairness only** — never scored and "
               "**never sent to the LLM**, so the model cannot build bias from them.")
    st.divider()
    if llm_available():
        st.success(f"LLM online · {LLM_MODEL}")
        st.caption(f"Endpoint: {LLM_BASE_URL}")
    else:
        st.info("LLM offline. Start Ollama locally (`ollama pull llama3.2`) or set LLM_BASE_URL/MODEL/API_KEY "
                "in Secrets to a free open-weight endpoint. The triage workflow runs without it.")

# ---------------------------------------------------------------- scored data
df, src = load_data()
scored = compute_scores(df, w)
eligible = scored[scored["eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
eligible["rank"] = eligible.index + 1
surfaced = eligible.head(SURFACE)
ineligible_n = int((~scored["eligible"]).sum())

st.title("Social Welfare Grant — Application Triage")
st.caption(f"Cycle 2026-Q3 · Officer console · **AI ranks, explains & supports; officers decide every grant.** · data: {src}")
c1, c2, c3 = st.columns(3)
c1.metric("Applications received", len(df), f"-{ineligible_n} filtered by eligibility")
c2.metric("AI-surfaced for review", min(SURFACE, len(eligible)), "ranked by Need Index")
c3.metric("Officer-approved grants", f"{selected_count()} / {TARGET}", f"{TARGET - selected_count()} remaining")

tab_q, tab_a, tab_f, tab_p, tab_log = st.tabs(
    ["📋 Review queue", "💬 Officer copilot", "⚖️ Fairness monitor", "🎚️ Scoring policy", "🧾 Audit trail"])

# ---------------------------------------------------------------- copilot helpers
STATUS_BADGE = {"selected": "🟢 Selected", "rejected": "🔴 Set aside", "pending": "🟡 Pending"}


def dataset_summary():
    by_inc = eligible["income_bracket"].value_counts().to_dict()
    by_wc = eligible["workclass"].value_counts().head(5).to_dict()
    def fair(dim):
        rows = []
        for gkey in sorted(eligible[dim].unique()):
            pool = eligible[eligible[dim] == gkey]; s = surfaced[surfaced[dim] == gkey]
            sel = sum(1 for i in pool["id"] if st.session_state.decisions.get(i, {}).get("status") == "selected")
            rows.append(f"{gkey}: pool {len(pool)}, surfaced {len(s)}, selected {sel}")
        return "; ".join(rows)
    return (f"Total {len(df)}; eligible {len(eligible)}; filtered {ineligible_n}. Surfaced top {len(surfaced)}. "
            f"Target {TARGET}; approved {selected_count()}. Income band counts {by_inc}. Top workclasses {by_wc}. "
            f"Fairness AGGREGATES (monitoring only) sex — {fair('sex')}; race — {fair('race')}. "
            f"Surfaced score range {int(surfaced['score'].min())}-{int(surfaced['score'].max())}.")


def retrieve(query):
    import re
    ids = [x.upper() for x in re.findall(r"app-\d+", query, flags=re.I)]
    hits = eligible[eligible["id"].isin(ids)]
    if len(hits) < 12:
        q = query.lower()
        mask = eligible.apply(lambda row: any(t in f"{row['income_bracket']} {row['workclass']} {row['occupation']}".lower()
                                              for t in q.split() if len(t) > 3), axis=1)
        hits = pd.concat([hits, eligible[mask]]).drop_duplicates("id")
    if len(hits) == 0:
        hits = surfaced.head(8)
    hits = hits.head(12)
    # records sent to the LLM contain NO PII and NO protected attributes
    lines = [f"{row['id']} | rank {int(row['rank'])} | score {int(row['score'])} | income {row['income_bracket']} | "
             f"hours {int(row['hours_per_week'])} | edu {int(row['education_num'])}y | work {row['workclass']} | "
             f"age {int(row['age'])} | capital {'none' if row['capital_gain']==0 else 'some'} | "
             f"decision {st.session_state.decisions.get(row['id'], {}).get('status', 'pending')}"
             for _, row in hits.iterrows()]
    return list(hits["id"]), "\n".join(lines)


def render_copilot(suffix="main", height=360):
    """Grounded data copilot. Reusable so it can live in the review queue and its own tab."""
    st.caption("Grounded in stored data only · cites surrogate IDs · **no PII / protected attributes** · never decides.")
    box = st.container(height=height, border=True)
    with box:
        if not st.session_state.chat:
            st.caption("💡 Try: *Why is APP-1000 ranked where it is?* · *How many surfaced applicants are below the threshold?*")
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.write(m["content"])
    if not llm_available():
        st.info("Start the LLM (see sidebar) to enable the copilot.")
        return
    prompt = st.chat_input("Ask the copilot…", key=f"chat_input_{suffix}")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        ids, records = retrieve(prompt)
        sys = ("You are the data copilot for a welfare officer during triage. Answer ONLY using the DATA CONTEXT below. "
               "If it is not there, say you don't have it — never invent applicants or numbers. Cite surrogate IDs. You "
               "may compute counts, comparisons and rankings. You must NOT make or recommend the final grant decision — "
               "only the officer decides; surface considerations, not verdicts. You are given NO PII and NO protected "
               "attributes at the record level; fairness figures are aggregates for monitoring only and must never be "
               "used to justify ranking one applicant above another. Be concise.\n\nDATASET SUMMARY\n" + dataset_summary()
               + "\n\nRETRIEVED RECORDS (no PII, no protected attributes)\n" + records)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-6:]]
        try:
            with st.spinner("Composing a grounded answer…"):
                ans = call_llm(sys, history, max_tokens=500)
            st.session_state.chat.append({"role": "assistant", "content": ans})
        except Exception:
            st.session_state.chat.append({"role": "assistant", "content": "⚠️ Couldn't reach the LLM. Please try again."})
        st.rerun()


# ---------------------------------------------------------------- review queue
with tab_q:
    # applicant picker across the top (compact)
    opts = [f"#{int(r['rank']):>3} · {int(r['score']):>3} pts · {r['id']}" for _, r in surfaced.iterrows()]
    choice = st.selectbox(f"📋 Review queue · top {len(surfaced)} by Need Index", opts)

    sel_id = choice.split(" · ")[2]
    r = surfaced[surfaced["id"] == sel_id].iloc[0]
    d = st.session_state.decisions.get(sel_id, {})
    status = d.get("status", "pending")
    badge = STATUS_BADGE[status]
    cap_txt = "None" if int(r["capital_gain"]) == 0 else f"{int(r['capital_gain']):,}"

    left, right = st.columns([1, 1], gap="large")

    # ============ LEFT: officer decision + applicant details + graph =====
    with left:
        with st.container(border=True):
            st.markdown("**🗳️ Officer decision**")
            at_cap = selected_count() >= TARGET and status != "selected"
            note = st.text_area("Note / override justification", value=d.get("note", ""), key=f"n_{sel_id}",
                                placeholder="e.g. Interview confirms severe hardship — prioritise despite mid rank.")
            b = st.columns(4)
            if b[0].button("✅ Approve", key=f"ap_{sel_id}", disabled=at_cap, use_container_width=True):
                st.session_state.decisions[sel_id] = {"status": "selected", "note": note}; log(sel_id, "Approved for final list", int(r["rank"]), int(r["score"]), note); st.rerun()
            if b[1].button("❌ Set aside", key=f"rj_{sel_id}", use_container_width=True):
                st.session_state.decisions[sel_id] = {"status": "rejected", "note": note}; log(sel_id, "Set aside", int(r["rank"]), int(r["score"]), note); st.rerun()
            if b[2].button("↩ Reset", key=f"rs_{sel_id}", use_container_width=True):
                st.session_state.decisions[sel_id] = {"status": "pending", "note": note}; log(sel_id, "Reset to pending", int(r["rank"]), int(r["score"])); st.rerun()
            if b[3].button("🚩 Flag", key=f"fl_{sel_id}", use_container_width=True):
                st.session_state.decisions[sel_id] = {**d, "status": status, "note": note, "flagged": True}; log(sel_id, "Flagged for supervisor", int(r["rank"]), int(r["score"])); st.rerun()
            if at_cap:
                st.warning("The 50-grant limit is reached. Set an approved applicant aside to free a place.")

        # header line
        st.markdown(f"**{r['id']}** · 🔒 `{r['pii_hash']}` &nbsp;|&nbsp; "
                    f"Score **{int(r['score'])}/100** · Rank **#{int(r['rank'])}** · {badge}")

        # applicant details — clear labels, readable medium font
        with st.container(border=True):
            st.markdown("**Applicant details**")
            def _field(label, value):
                return (f"<div style='font-size:0.95rem; line-height:1.4; margin-bottom:6px'>"
                        f"<span style='color:#6b7280'>{label}</span><br>"
                        f"<b style='font-size:1.05rem'>{value}</b></div>")
            d1, d2, d3 = st.columns(3)
            d1.markdown(_field("Income band", r["income_bracket"]), unsafe_allow_html=True)
            d2.markdown(_field("Work hours", f"{int(r['hours_per_week'])}/wk"), unsafe_allow_html=True)
            d3.markdown(_field("Education", f"{int(r['education_num'])} yrs"), unsafe_allow_html=True)
            d4, d5, d6 = st.columns(3)
            d4.markdown(_field("Age", int(r["age"])), unsafe_allow_html=True)
            d5.markdown(_field("Work type", r["workclass"]), unsafe_allow_html=True)
            d6.markdown(_field("Capital income", cap_txt), unsafe_allow_html=True)

        # the graph — why this applicant ranked here
        with st.container(border=True):
            st.markdown("**📊 Why the rubric ranked this applicant** (points / 100)")
            contrib = {k: float(r[f"c::{k}"]) for k in w}
            st.bar_chart(pd.Series(contrib).sort_values(ascending=True), horizontal=True, height=240)
            st.caption("Reasons: " + " · ".join(plain_reasons(r)))
            st.caption("🔒 Sex, race and origin are monitored for fairness only — never scored, ranked or sent to the LLM.")

        with st.expander("📄 De-identified statement (the only applicant text the LLM sees)"):
            st.write(r["narrative"])

    # ============ RIGHT: advisory + copilot (highlighted) ================
    with right:
        with st.container(border=True):
            st.markdown("**🟣 LLM advisory read** · optional — does not set the score or decide")
            if not llm_available():
                st.caption("Start the LLM (see sidebar) to enable this.")
            elif st.button("Generate LLM assessment", key=f"a_{sel_id}", use_container_width=True):
                sys = ("You support a welfare officer by reading a DE-IDENTIFIED applicant statement and non-identifying "
                       "facts, then giving a QUALITATIVE, ADVISORY read to speed up review. You do NOT set the numeric score "
                       "and do NOT make or recommend the final grant decision — the officer decides. You are given no names, "
                       "IDs, sex, race or origin; never ask for or infer them. In <=110 words give: (1) a 2-sentence read of "
                       "how strongly this profile matches the program's need criteria — low / moderate / high (advisory only); "
                       "(2) up to 3 points to verify; (3) up to 2 questions the officer could ask.")
                try:
                    with st.spinner("Reading the statement…"):
                        txt = call_llm(sys, [{"role": "user", "content": f'Statement: """{r["narrative"]}"""\nFacts: {llm_safe_facts(r)}'}])
                    st.info(txt)
                    log(sel_id, "LLM advisory read generated", int(r["rank"]), int(r["score"]))
                except Exception:
                    st.error("Couldn't reach the LLM. Check the sidebar status and try again.")

        with st.container(border=True):
            st.markdown("### 💬 Officer copilot")
            render_copilot(suffix="queue", height=300)


# ---------------------------------------------------------------- officer copilot (full tab)
with tab_a:
    st.markdown("#### 💬 Officer copilot")
    st.markdown("Ask about rankings, counts, comparisons or fairness. The copilot answers **only from the stored data**, "
                "cites surrogate IDs, uses **no PII or protected attributes**, and never makes the final decision.")
    render_copilot(suffix="tab", height=480)


# ---------------------------------------------------------------- fairness
with tab_f:
    st.markdown("#### Disparate-impact monitor")
    st.caption("Protected attributes never enter the score or the LLM. We watch outcomes instead. The four-fifths rule "
               "flags a group whose selection rate falls below 80% of the strongest group's.")
    for dim, title in [("sex", "By sex"), ("race", "By race")]:
        rows = []
        for gkey in sorted(eligible[dim].unique()):
            pool = eligible[eligible[dim] == gkey]; s = surfaced[surfaced[dim] == gkey]
            sel = sum(1 for i in pool["id"] if st.session_state.decisions.get(i, {}).get("status") == "selected")
            rows.append({"Group": str(gkey), "Surfaced %": round(100 * len(s) / max(1, len(pool)), 1),
                         "Selected %": round(100 * sel / max(1, len(pool)), 1), "Pool": len(pool)})
        fdf = pd.DataFrame(rows).set_index("Group")
        st.markdown(f"**{title}**")
        st.bar_chart(fdf[["Surfaced %", "Selected %"]], height=240)
        m = fdf["Selected %"].max()
        if m > 0:
            flagged = fdf[fdf["Selected %"] < 0.8 * m]
            if len(flagged):
                st.error("Four-fifths rule triggered for: " + ", ".join(flagged.index) + ". Review before finalising.")
            else:
                st.success("No four-fifths disparity detected in current selections.")
        else:
            st.caption("Selection rates appear once you approve applicants.")


# ---------------------------------------------------------------- policy
with tab_p:
    st.markdown("#### Need-index weights (policy-controlled)")
    st.caption("Set in the sidebar. Changing them instantly re-ranks the queue, so the agency sees the effect of any "
               "criteria change and the logic stays auditable.")
    s = pd.Series(w); s = (s / s.sum() * 100).round(1)
    st.bar_chart(s.sort_values(), horizontal=True, height=260)
    st.success("Interpretable, not a black box: a weighted rubric can be published, contested and appealed. Every score "
               "decomposes into named factors an officer and a citizen can understand.")


# ---------------------------------------------------------------- audit
with tab_log:
    st.markdown("#### Decision audit trail")
    st.caption("Every LLM and officer action is timestamped and attributable.")
    if st.session_state.audit:
        adf = pd.DataFrame(st.session_state.audit)
        st.dataframe(adf, hide_index=True, use_container_width=True, height=320)
        final = eligible[eligible["id"].isin([i for i, d in st.session_state.decisions.items() if d.get("status") == "selected"])]
        cols = st.columns(2)
        cols[0].download_button("⬇ Final 50 (CSV)", final[["rank", "id", "score", "income_bracket", "hours_per_week", "education_num"]].to_csv(index=False), "final_selection.csv", "text/csv")
        cols[1].download_button("⬇ Audit trail (CSV)", adf.to_csv(index=False), "audit_trail.csv", "text/csv")
    else:
        st.info("No actions recorded yet. Decisions in the review queue appear here.")

st.caption("Prototype · open data + a free open-weight LLM · AI assists, explains and answers; officers review, may "
           "override, and make every final decision. No PII or protected attributes are ever sent to the LLM.")
