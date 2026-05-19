# Clinical Trial Similarity & Outcome Analysis Tool

An NLP-powered tool that identifies similar clinical trials, links them to their published research papers, and summarizes key outcomes — all from a single NCT ID.

Built using data from [ClinicalTrials.gov](https://clinicaltrials.gov) (563,000+ trials) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/).

---

## What It Does

Given an NCT ID, the tool:

1. **Finds the top 10 most similar trials** using a weighted combination of study design, text, and outcome similarity
2. **Links each trial to its PubMed publication** using a three-strategy matching approach with confidence scoring
3. **Summarizes key findings** from publication abstracts using either rule-based extraction or AI-powered summarization (Gemini)

Optional filters let you restrict results to trials with the **same drug/intervention** or **same disease/condition**.

---

## Demo

> Input: `NCT04458610` with same-drug and same-disease filters enabled

The tool returns the query trial with its linked publication and outcome summary, followed by ranked similar trials — each showing how similarity breaks down across design, text, and outcomes, plus any linked publications found.

---

## How Similarity Works

Similarity is computed using cosine similarity across three separate feature dimensions:

| Component | Method | Default Weight |
|-----------|--------|---------------|
| Design similarity | Cosine similarity on one-hot encoded study design features | 50% |
| Text similarity | Cosine similarity on TF-IDF vectors (vocab = 500) | 20% |
| Outcome similarity | Cosine similarity on outcome TF-IDF vectors (vocab = 100) | 30% |

**Combined Score** = `0.5 × Design + 0.2 × Text + 0.3 × Outcome`

### Why TF-IDF?

TF-IDF was selected because it naturally upweights rare but meaningful medical terms (specific drug names, biomarkers, disease subtypes) while downweighting generic clinical language that appears in nearly every trial. This ensures similarity is driven by distinctive scientific concepts rather than common phrasing.

The method also scales efficiently to 500,000+ trials. TF-IDF produces sparse vectors enabling fast cosine similarity computation, whereas dense embedding models like BERT would require significantly more memory and slower lookup times for real-time search across the full corpus.

Additionally, TF-IDF offers interpretability—researchers can examine which specific terms drove a similarity score, validating that matches reflect meaningful relationships. And because it's unsupervised, it requires no labeled training data of similar trial pairs, which don't exist for this domain.

---

## PubMed Linking

Three strategies are tried in order, with explicit confidence scores so you know how much to trust each match:

| Strategy | Confidence | Method |
|----------|-----------|--------|
| Direct NCT ID search | 95% | Searches PubMed for articles explicitly mentioning the NCT identifier |
| Fuzzy title matching | up to 75% | Extracts key terms from the trial title, applies token-set-ratio fuzzy matching |
| Condition/intervention search | up to 60% | Queries PubMed via MeSH terms and title/abstract, scores by content overlap |

If no strategy reaches the confidence threshold, the tool returns "No linked publication found" rather than showing a low-confidence guess.

---

## Outcome Summarization

Two summarization methods are available:

| Method | Quality | Cost | Requirements |
|--------|---------|------|--------------|
| Rule-based | Good | Free | None |
| Gemini AI | Better | ~$0.01 per 1000 abstracts | Gemini API key |

**Rule-based** (default): Scores sentences based on importance keywords (e.g., "significant", "effective", "conclude", "primary endpoint") and statistical indicators (p-values, hazard ratios). Position weighting favors conclusion sentences typically found at the end of abstracts.

**Gemini AI** (optional): Sends abstracts to Google's Gemini 2.5 Flash Lite model with a prompt optimized for clinical outcome extraction. Produces more coherent, contextual summaries. Extremely cost-effective at ~$0.075 per million input tokens.

---

## Project Structure

```
.
├── bulk_download_all.py          # Downloads all trials from ClinicalTrials.gov API
├── API_study_grabber.py          # Lightweight single-study API wrapper
├── similarity.py                 # Core TF-IDF similarity engine + index builder
├── pubmed_connector.py           # PubMed matching + outcome summarization (rule-based & Gemini)
├── similarity_flask_enhanced.py  # Flask REST API backend
├── interface_enhanced.html       # Frontend (vanilla HTML/CSS/JS, no build step)
├── .env                          # API keys (create from .env.example, not committed)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Excludes .env, data files, and caches
└── all_trials_data/              # Downloaded trial data (generated, not committed)
    └── all_trials_complete.json
```

---

## Setup

### Requirements

- Python 3.8+
- ~4 GB RAM for index building (3 GB at peak with sequential processing)
- ~8 GB disk space for the full trial dataset and index

### Install dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy requests fuzzywuzzy python-Levenshtein python-dotenv google-genai
```

### Step 1 — Create your `.env` file

Create a file named `.env` in your project root with your API keys:

```bash
# =============================================================================
# PUBMED / NCBI CONFIGURATION
# =============================================================================

# Your email (required by NCBI terms of service)
PUBMED_EMAIL=your_email@example.com

# NCBI API Key (optional, but increases rate limit from 3 to 10 requests/sec)
# Get your free key at: https://www.ncbi.nlm.nih.gov/account/settings/
PUBMED_API_KEY=your_ncbi_api_key_here

# =============================================================================
# GEMINI CONFIGURATION (for AI-powered outcome summarization)
# =============================================================================

# Google Gemini API Key (optional - falls back to rule-based if not set)
# Get your free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:** Add `.env` to your `.gitignore` to avoid committing your API keys:

```bash
echo ".env" >> .gitignore
```

### Step 2 — Download the trial data

```bash
python bulk_download_all.py
```

Select option `1` to start a fresh download. This paginates through all ~563,000 trials from ClinicalTrials.gov in batches of 1,000 (the API maximum). Expect **2–4 hours** for the full download.

Checkpoints are saved every 10,000 trials to `all_trials_data/checkpoint_page_N.json`. If the download is interrupted, run the script again and select option `3` to resume from the latest checkpoint.

### Step 3 — Build the similarity index

```bash
python bulk_download_all.py
```

Select option `2` and provide the path to `all_trials_data/all_trials_complete.json`. Index building takes **20–40 minutes** and saves `trial_similarity_index_complete.pkl`.

> Both steps can be done in one go with option `1`.

### Step 4 — Start the API server

```bash
python similarity_flask_enhanced.py
```

The server starts on `http://localhost:5001`. You should see:

```
✓ Loaded 563,083 trials
✓ PubMed connector initialized (email: your_email@example.com)
✓ Outcome summarizer initialized (Gemini AI)
```

If you don't have a Gemini API key configured, you'll see:

```
✓ Outcome summarizer initialized (rule-based)
  Tip: Set GEMINI_API_KEY env var for AI-powered summaries
```

### Step 5 — Open the frontend

Open `interface_enhanced.html` in your browser. No web server needed — it communicates directly with the Flask backend.

---

## API Keys

### NCBI / PubMed API Key (Recommended)

Increases PubMed rate limits from 3 to 10 requests/second.

1. Create an account at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/)
2. Go to Account Settings
3. Scroll to "API Key Management" and click "Create an API Key"
4. Copy the key to your `.env` file

### Gemini API Key (Optional)

Enables AI-powered outcome summarization. Extremely cheap (~$0.01 per 1000 abstracts).

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key to your `.env` file

The free tier has rate limits. For production use, add billing to your Google Cloud account.

---

## Limitations

- **PubMed linking** is not guaranteed — many older publications do not include the NCT identifier. Confidence scores indicate match reliability.
- **Outcome summarization** quality depends on the method used. Rule-based extraction selects existing sentences and can be misled by background sentences. Gemini produces better summaries but requires an API key.
- **Paywalled publications** cannot be accessed — outcome summaries are drawn from publicly available abstracts only.
- **Index freshness** — the index is a snapshot of ClinicalTrials.gov at download time. Rebuild periodically to capture new registrations.
- **Memory** — index building requires ~3 GB RAM peak and roughly 2–5 hours total. Not suitable for machines with less than 8 GB RAM.

---

## Future Work

- Semantic similarity via SBERT sentence embeddings as a supplement to TF-IDF
- Trial acronym resolution to improve PubMed linking for major trial programs
- Automated incremental index updates synced with ClinicalTrials.gov
- User-configurable similarity weight profiles saved per researcher
- Support for additional LLM providers (OpenAI, Anthropic) for outcome summarization

---

## Data Sources

- **ClinicalTrials.gov** — Trial data retrieved via the [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api). Public domain.
- **PubMed** — Publication data retrieved via [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/). Subject to NCBI terms of use.
