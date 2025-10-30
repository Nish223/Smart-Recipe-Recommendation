# Smart Recipe Recommendation Engine

A lightweight Streamlit app that helps you search and explore recipes by ingredients or dish name, filter by simple heuristics (diet, flavor, meal type, prep time), suggest ingredient substitutes, recommend similar recipes using TF-IDF, and show approximate nutrition totals.

This repository is intended as a self-contained demo for an ingredient-focused recipe search & recommendation prototype. It purposely avoids heavy native dependencies (e.g., `gensim` C-extensions) so it runs easily on most machines.

---

## Features

- Ingredient normalization and cleaning
- Search by ingredients (substring matching) or dish name
- Filters: Diet (veg/non-veg/vegan heuristic), Flavor (sweet/savory), Meal type, Prep time
- Recipe recommendations using TF-IDF + cosine similarity
- Ingredient substitution suggestions using a lightweight co-occurrence/token-overlap fallback
- Directions summarization (safe wrapper that uses transformers if installed; otherwise falls back to first few steps)
- Approximate nutrition aggregation from a small ingredient mapping

> Note: Cuisine prediction (`cuisine_model.pkl`) has been removed from the main app for simplicity. A training script is provided (optional) if you later add cuisine labels to your dataset.

---

## Files to upload to GitHub

Include the following files in the repository root:

- `app.py` — Streamlit application (main entrypoint)
- `utils.py` — Helper functions (data loading, normalization, filters, TF-IDF recommender, substitutes, summarization wrapper)
- `nutrition.py` — Small nutrition lookup + aggregator
- `train_cuisine.py` *(optional)* — Script to train a cuisine classifier (only if you have `cuisine` labels)
- `full_dataset.csv` (or your real dataset CSV) — REQUIRED: your recipe CSV file. Make sure to use the same filename referenced in `app.py` (default: `full_dataset.csv` or edit `DATA_PATH`).
- `nb1.ipynb`, `nb2.ipynb` *(optional)* — Your exploratory notebooks (if you want to include them).
- `README.md` — This file.
- `requirements.txt` — list of Python dependencies.
- `.gitignore` — recommended to exclude virtual environments, large models, etc.

**Do NOT upload:**
- Large model files (e.g., `cuisine_model.pkl`) unless necessary (these increase repo size).
- Virtual environment folders (e.g., `venv/`).
- Data that is private or large (>100 MB). Use a link or Git LFS if needed.

---

## Quick start (developer)

### 1. Clone repo
```bash
git clone <your-repo-url>
cd <repo-name>

2. (Recommended) Create & activate a virtual environment

Windows (PowerShell):

python -m venv venv
venv\Scripts\activate


macOS / Linux:

python -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


Note: transformers and torch are optional — the app will fall back to a simple summary if they are not present. If you want summarization using BART/T5, keep transformers and torch in the environment (they can be large).

4. Dataset
The full dataset (`full_dataset.csv`) is too large to upload on GitHub.
Place it manually in the project root folder before running the app.

Steps
1. Download the dataset from https://www.kaggle.com/datasets/saldenisov/recipenlg .
2. Save it as `full_dataset.csv` in the same folder as `app.py`.
3. Then run:
   ```bash
   streamlit run app.py

