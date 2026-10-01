# Solution design — detailed
### How each stage works, the prompts, the criteria, and the privacy & fairness controls

The system is a **decision-support copilot**, not a decision-maker. A free open-weight LLM (Ollama) makes officer review faster and better-informed; a transparent rubric produces the score; and the officer makes every final grant decision. Two rules hold at every stage: **(1) the LLM never sets the score or makes the decision; (2) no PII and no protected attribute ever reaches the LLM.**

Pipeline: **Intake & mask → Eligibility gate → Need Index ranking → Surface top 100 → Officer review (LLM copilot) → Officer approves 50**, with Fairness monitoring and an Audit trail running across all of it.

---

## Stage 1 — Intake & PII masking

**Purpose.** Load applicant records and make them privacy-safe *before* anything else touches them.

**What happens (see `prepare_data.py`).**
- Direct identifiers are **never stored**. Each record gets a surrogate id (`APP-1000`) and a salted, truncated hash (`pii_hash = sha256(salt + raw_id)[:12]`) that stands in for any real identifier. Officers only ever see the surrogate id.
- **Protected attributes** (sex, race, origin) are placed in segregated columns, flagged as monitoring-only.
- A **de-identified narrative** is generated from need-relevant features — containing no name, no id, and no protected term. This is the *only* applicant text the LLM is ever given.
- A verification step asserts the narrative contains none of a banned list of protected/PII terms; the build fails if it does.

**Inputs → outputs.** Raw open dataset (UCI Adult, or the Kaggle IDB Costa Rican Household Poverty set) → `applicants.csv` with surrogate ids, masked hash, segregated protected columns, and a clean narrative.

**Responsible-AI / privacy.** This stage is where the privacy guarantee is established: masking at storage prevents identifiers from leaking downstream, and segregating protected attributes means they can be *monitored* without ever being *used*.

---

## Stage 2 — Eligibility gate (hard rules, not AI)

**Purpose.** Remove applications that are categorically out of scope, using deterministic policy rules that are easy to audit and cannot be "learned away" by a model.

**Filtering criteria (illustrative; set with the agency).**
- Must be a resident (residency proxy).
- Not clearly above the means test — e.g. income band `>50K` **and** capital gains `> 7000` → ineligible (financially secure).

Everything that passes the gate goes forward for scoring; everything filtered is logged with its reason. Keeping this as explicit rules — separate from the Need Index — means an applicant can be told precisely why they were ruled ineligible.

**Responsible-AI.** Hard rules are transparent and contestable. No ML and no LLM are involved here by design.

---

## Stage 3 — Need Index ranking (transparent, interpretable)

**Purpose.** Score every eligible application for need on a published, policy-weighted rubric — *not* a black box — so each score decomposes into named factors an officer and a citizen can understand.

**Formula.** `score = 100 × Σ(weightᵢ × factorᵢ) / Σ(weightᵢ)`. Each factor is normalised to 0–1 from non-protected features.

**Criteria & default weights** (editable live in the app; set by policy with the agency):

| Factor | Signal (from the data) | Weight |
|---|---|---|
| Low income | income band at/below the means-test threshold; no capital income | 34 |
| Underemployment | weekly hours below full time | 20 |
| Limited education | years of schooling below 12 | 14 |
| Precarious work | unpaid / never-worked / informal work type | 14 |
| Age vulnerability | elderly or very young | 10 |
| No capital buffer | no reported capital/asset income | 8 |

**Why a rubric and not an LLM score.** A weighted rubric is publishable, contestable, and appealable; it produces the same score every time; and it avoids learning historical bias from past officer decisions. An LLM assigning the priority number would be opaque, inconsistent on re-runs, and able to pick up subtle demographic signals — all unacceptable for a legal entitlement. **Protected attributes are not inputs.**

**Output.** Each eligible record gets a `score` (0–100), a per-factor contribution breakdown (the explainability bars in the UI), and a rank.

---

## Stage 4 — Surface the top 100

**Purpose.** Convert 500 applications into a review queue sized to the officers' real time budget. The AI **surfaces** the highest-need 100; the other 400 remain available but de-prioritised. This is the single biggest efficiency lever — officers read 100, not 500 (~80% less screening volume) — and it is still just a prioritisation, not a decision.

---

## Stage 5 — Officer review with the LLM copilot (the leverage point)

This is where the LLM earns its place. It does four things, **all advisory**, over the de-identified data only.

### 5a. LLM advisory read (narrative assessment)
Reads the de-identified statement + non-identifying facts and returns a qualitative read to speed up review.

**System prompt (verbatim, from `app.py`):**
> You support a welfare officer by reading a DE-IDENTIFIED applicant statement and non-identifying facts, then giving a QUALITATIVE, ADVISORY read to speed up review. You do NOT set the numeric score and do NOT make or recommend the final grant decision — the officer decides. You are given no names, IDs, sex, race or origin; never ask for or infer them. In ≤110 words give: (1) a 2-sentence read of how strongly this profile matches the program's need criteria — low / moderate / high (advisory only); (2) up to 3 points to verify; (3) up to 2 questions the officer could ask.

**User message:** the de-identified `narrative` plus `llm_safe_facts()` — income band, hours, schooling years, work type, age band, capital-income flag. **No** name, id, sex, race, or origin is ever included.

**How it helps the decision.** It turns free text into a structured read, gives a non-binding low/moderate/high need-match, flags what to verify, and suggests questions — so the officer reviews faster and with better context, while still deciding.

### 5b. Officer actions (human-in-the-loop)
The officer can **approve** (toward the 50), **set aside**, **reset**, or **flag for supervisor**, each with a free-text justification. Overriding the AI rank is explicitly supported — a mid-ranked case can be approved (e.g. an interview reveals hardship the data missed) and a top-ranked one set aside. A **hard cap of 50** is enforced. Every action is written to the audit trail.

---

## Stage 6 — Officer copilot (RAG-grounded Q&A)

**Purpose.** Let officers ask natural-language questions about the cycle — rankings, counts, comparisons, fairness — answered strictly from the stored data.

**Retrieval (RAG).** The question is used to pull the most relevant records (by surrogate id or attribute keywords) plus a dataset summary. The records passed to the model contain **no PII and no protected attributes**; fairness figures are included only as aggregates for monitoring.

**System prompt (verbatim, from `app.py`):**
> You are the data copilot for a welfare officer during triage. Answer ONLY using the DATA CONTEXT below. If it is not there, say you don't have it — never invent applicants or numbers. Cite surrogate IDs. You may compute counts, comparisons and rankings. You must NOT make or recommend the final grant decision — only the officer decides; surface considerations, not verdicts. You are given NO PII and NO protected attributes at the record level; fairness figures are aggregates for monitoring only and must never be used to justify ranking one applicant above another. Be concise.

Then a `DATASET SUMMARY` block and a `RETRIEVED RECORDS (no PII, no protected attributes)` block are appended. The retrieval trace shown in the UI ("Retrieved: summary + N records — APP-…") makes the grounding visible.

**Production upgrade.** Swap the keyword retriever for embeddings + a vector store, and give the copilot explicit tools (`query_applications`, `aggregate_stats`, `get_score_breakdown`, `compute_fairness`) so arithmetic is exact and every answer is traceable to a tool result — with the same guardrails.

---

## Fairness monitoring (runs across the pipeline)

Because protected attributes never enter the score or the LLM, bias can only show up in **outcomes** — so we watch outcomes. For each protected group the monitor computes the share surfaced into the top 100 and the share selected, and applies the **four-fifths (80%) rule**: if any group's selection rate falls below 80% of the strongest group's, it raises a flag for the agency to investigate *before* finalising. This catches proxy discrimination even with no protected attribute in the model.

---

## Audit trail & export

Every LLM read, every officer action, and every override justification is timestamped and attributed, stored alongside the AI rank and score. The final 50 and the full audit log export to CSV. This makes the whole cycle reconstructable for internal audit, oversight, and appeals.

---

## Privacy & PII — the guarantee, in one place

- **Masked at storage.** Direct identifiers are hashed (salted SHA-256); only surrogate ids are shown. The stored file contains no names.
- **Never sent to the LLM.** The model receives only the de-identified narrative + non-identifying facts. A build-time check verifies the narrative is free of PII/protected terms.
- **Protected attributes monitored, never used.** Sex, race and origin live in segregated columns used *only* in the fairness monitor — never in scoring, never in a prompt. This means the model **cannot learn or amplify bias** from them, and the ranking cannot be driven by them.
- **Decision boundary.** No AI component can grant a benefit; the officer confirms every one.

---

## Key questions for the agency

- Which criteria are **hard** (eligibility) vs **soft** (priority), and is there an official weighting or statutory priority order?
- The agency's working **definition of fairness** — equal treatment of like cases, proportional representation, or prioritising the worst-off? (They can conflict; the choice drives the metric and the four-fifths threshold.)
- The **appeals** process, and what an applicant must be told about how they were assessed.
- Which fields are **verified** at intake, what is typically missing, and how often a cycle runs (this sizes the "surface 100").
- Any mandatory **quotas or set-asides** (e.g. disability, geographic spread).
- **Data protection** obligations — retention, residency, and who may see what.

---

## How this maps to the evaluation criteria

- **AI solution approach** — a clear intake→mask→gate→score→surface→review pipeline, with a defensible interpretable scorer and an LLM used precisely where it adds value.
- **Presentation & storytelling** — the 500 → 100 → 50 funnel and the "LLM assists, officer decides" line carry both slides and the live app.
- **Responsible AI** — PII masking, protected-attribute exclusion, disparate-impact monitoring, no bias feedback loop, data minimisation, and a human decision boundary.
- **Explainability & human oversight** — per-factor score breakdowns, plain-language reasons, an advisory (not authoritative) LLM, officer override with justification, a hard cap, and a complete audit trail.
