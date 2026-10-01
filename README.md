# Welfare Grant Triage — Streamlit prototype (open data + free open-weight LLM)

AI-assisted triage for a social welfare grant: open applicant data → PII masking → eligibility gate → a transparent **Need Index** ranks and **surfaces the top 100** → officers review (with an LLM copilot) and **approve the final 50**. The LLM is a **free, open-weight model via Ollama** and acts only as decision *support* — it never sees PII and never makes the decision.

## 1. Get the data (open source)
```bash
python prepare_data.py --source adult  --n 500      # UCI Adult / Census Income (open, no login)
# or the on-topic welfare set: download the Kaggle IDB "Costa Rican Household Poverty" train.csv, then:
python prepare_data.py --source kaggle --path train.csv --n 500
# or fully offline (simulated, same schema):
python prepare_data.py --source sample --n 500
```
This writes `applicants.csv`. The script **masks PII**: no names/identifiers are stored (only a surrogate `APP-id` + a salted hash), protected attributes are segregated for fairness monitoring, and the de-identified `narrative` (the only text the LLM sees) is verified free of names, IDs and protected terms.

**Datasets**
- UCI Adult / Census Income — the standard algorithmic-fairness dataset: https://archive.ics.uci.edu/dataset/2/adult
- IDB Costa Rican Household Poverty (most on-topic): https://www.kaggle.com/c/costa-rican-household-poverty-prediction

## 2. Run the free LLM (Ollama)
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2        # small & free; or qwen2.5:3b, phi3.5, mistral
```
Ollama serves an OpenAI-compatible API at `http://localhost:11434/v1`, which the app uses by default. No key, no cost, runs offline.

## 3. Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Host a shareable URL (Streamlit Community Cloud — free)
1. Put `app.py`, `requirements.txt` and `applicants.csv` in a **public GitHub repo**.
2. Deploy at https://share.streamlit.io (pick the repo + `app.py`). You get a `*.streamlit.app` URL.
3. **The triage workflow runs fully without any LLM.** A *local* Ollama is not reachable from the cloud, so to enable the AI copilot on a hosted URL, point the app at any free OpenAI-compatible open-weight endpoint via the app's **Secrets**:
   ```toml
   LLM_BASE_URL = "https://api.groq.com/openai/v1"   # free tier, serves Llama 3.x etc.
   LLM_MODEL    = "llama-3.1-8b-instant"
   LLM_API_KEY  = "gsk_..."
   ```
   (OpenRouter and other OpenAI-compatible hosts work the same way — just change the three values.)

## Responsible-AI & privacy design
- **PII never reaches the LLM.** Identifiers are hashed at storage; officers see surrogate IDs; the model receives only a de-identified statement + non-identifying facts.
- **Protected attributes** (sex, race, origin) are **monitored for fairness only** — never scored, never sent to the model — so the model cannot learn or amplify bias from them.
- **Interpretable scoring, not a black box** — a published, policy-weighted rubric that can be explained, contested and appealed. The LLM does **not** set the score.
- **Human-in-the-loop** — officers override with justification, a hard 50-cap is enforced, every action is logged, and the final 50 + audit log export to CSV.
