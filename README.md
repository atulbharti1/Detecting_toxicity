# Twitch Toxicity Classifier

Binary toxicity classifier for political livestream chat comments on Twitch, built for a Kaggle competition as part of an ML course at Hertie School.

## Competition

[Predicting Toxicity in Twitch Comments](https://www.kaggle.com/competitions/predicting-toxicity-in-twitch-comments/overview)

The goal is to predict whether a Twitch chat comment is toxic (`IS_TOXIC = 1`) or not (`IS_TOXIC = 0`), based on data from 49 U.S. political livestream channels collected over August–September 2023 (~175,000 comments, ~12,000 unique users).

**Key challenges:**
- Class imbalance (~7% toxic)
- Temporal generalization — test set comes from later streams not seen during training
- Rich text data combined with structured metadata features

## Data

Download the data directly from the Kaggle competition page:
👉 [https://www.kaggle.com/competitions/predicting-toxicity-in-twitch-comments/data](https://www.kaggle.com/competitions/predicting-toxicity-in-twitch-comments/data)

Once downloaded, place the files in the `data/` folder:
```
data/
├── train.csv
└── test.csv
```

> `data/` is gitignored and not included in this repository.

## Approach

Built an ensemble of multiple models to handle the class imbalance and generalize across unseen livestream sessions. Due to hardware constraints, the ensemble is based on CPU-friendly models with text and metadata features rather than GPU-intensive approaches like BERT.

## Project Structure
```
twitch_toxicity/
├── data/                  # Raw data (not committed)
├── notebooks/
│   ├── baseline.ipynb     # Initial baseline model
│   ├── XGBoost.ipynb      # XGBoost experiments
│   └── baseline_BERT.ipynb
├── outputs/               # Submission CSV files
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Then open any notebook in the `notebooks/` folder to explore the models.

## Evaluation

Submissions are scored on **accuracy** (proportion of comments correctly classified).

## Approach

**Validation strategy:** Used `GroupShuffleSplit` with `livestream_id` as the grouping
key, ensuring no livestream session appears in both train and validation sets. This
mirrors the competition's session-based train/test split and avoids data leakage.

**Feature engineering:**
- *Regex features* — pattern-based signals from raw text (e.g. caps, URLs, mentions, punctuation)
- *Datetime features* — hour of day, day of week, and weekend indicator extracted from timestamps
- *TF-IDF* — light text preprocessing followed by TF-IDF vectorization to capture word-level signals

All three feature sets are combined via `scipy.sparse.hstack` into a single feature matrix,
with regex and datetime features scaled before combining.

**Model:** Logistic Regression with L2 regularization — a strong baseline for high-dimensional
sparse text features that is fast, interpretable, and resistant to overfitting.
git add README.md
git commit -m "Update README with project description"
git push origin main

