# Clinical Trial Similarity & Outcome Analysis Tool

An NLP-powered tool that identifies similar clinical trials, links them to their published research papers, and summarizes key outcomes — all from a single NCT ID.

Built using data from [ClinicalTrials.gov](https://clinicaltrials.gov) (563,000+ trials) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/).

---

## What It Does

Given an NCT ID, the tool:

1. **Finds the top 10 most similar trials** using a weighted combination of study design, text, and outcome similarity
2. **Links each trial to its PubMed publication** using a three-strategy matching approach with confidence scoring
3. **Summarizes key findings** extracted from publication abstracts — efficacy results, safety signals, and statistical conclusions

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

Weights can be overridden per search query.

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

## Project Structure

```
.
├── bulk_download_all.py          # Downloads all trials from ClinicalTrials.gov API
├── API_study_grabber.py          # Lightweight single-study API wrapper
├── similarity.py                 # Core TF-IDF similarity engine + index builder
├── pubmed_connector.py           # PubMed matching + outcome summarization
├── similarity_flaskv2.py         # Flask REST API backend
├── interface.html                # Frontend (vanilla HTML/CSS/JS, no build step)
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
pip install flask flask-cors scikit-learn pandas numpy requests fuzzywuzzy python-Levenshtein
```

### Step 1 — Download the trial data

```bash
python bulk_download_all.py
```

Select option `1` to start a fresh download. This paginates through all ~563,000 trials from ClinicalTrials.gov in batches of 1,000 (the API maximum). Expect **2–4 hours** for the full download.

Checkpoints are saved every 10,000 trials to `all_trials_data/checkpoint_page_N.json`. If the download is interrupted, run the script again and select option `3` to resume from the latest checkpoint.

### Step 2 — Build the similarity index

```bash
python bulk_download_all.py
```

Select option `2` and provide the path to `all_trials_data/all_trials_complete.json`. Index building takes **20–40 minutes** and saves `trial_similarity_index_complete.pkl`.

> Both steps can be done in one go with option `1`.

### Step 3 — Start the API server

```bash
python similarity_flaskv2.py
```

The server starts on `http://localhost:5001`.

### Step 4 — Open the frontend

Open `interface.html` in your browser. No web server needed — it communicates directly with the Flask backend.

---

## Configuration

Set your NCBI API key as an environment variable to increase PubMed rate limits from ~3 to 10 requests/second:

```bash
export PUBMED_EMAIL=your@email.com
export PUBMED_API_KEY=your_ncbi_api_key
```

You can get a free NCBI API key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).

---

## Limitations

- **PubMed linking** is not guaranteed — many older publications do not include the NCT identifier. Confidence scores indicate match reliability.
- **Outcome summarization** is rule-based and extractive. It selects existing sentences from the abstract rather than generating new text. It can be misled by background sentences that use result-adjacent language.
- **Paywalled publications** cannot be accessed — outcome summaries are drawn from publicly available abstracts only.
- **Index freshness** — the index is a snapshot of ClinicalTrials.gov at download time. Rebuild periodically to capture new registrations.
- **Memory** — index building requires ~3 GB RAM peak and roughly 2–5 hours total. Not suitable for machines with less than 8 GB RAM.

---

## Future Work

- Transformer-based summarization (BioBART, Clinical-T5) for higher-quality outcome extraction
- Semantic similarity via SBERT sentence embeddings as a supplement to TF-IDF
- Trial acronym resolution to improve PubMed linking for major trial programs
- Automated incremental index updates synced with ClinicalTrials.gov
- User-configurable similarity weight profiles saved per researcher

---

## Data Sources

- **ClinicalTrials.gov** — Trial data retrieved via the [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api). Public domain.
- **PubMed** — Publication data retrieved via [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/). Subject to NCBI terms of use.

