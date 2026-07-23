# Smart MCQ Solver Challenge

**IITM BS Degree — Deep Learning & Generative AI**
Name: Vatsal Gupta · Roll Number: 23F3002118

## Problem Statement

This project solves the Kaggle "Smart MCQ Solver Challenge": given a prompt
and five candidate options (A–E), predict the top-3 most likely correct
options, evaluated using MAP@3 (Mean Average Precision at 3).

## Repository Structure

```
smart-mcq-solver/
├── notebooks/       # Original Kaggle notebook (EDA, training, submission)
├── src/             # Reusable, script-based version of the pipeline
│   ├── utils.py     # Data loading, preprocessing, datasets, metrics, model classes
│   ├── train.py     # Trains TF-IDF, BiLSTM+Attention, GRU, and BERT
│   └── inference.py # Loads checkpoints, generates submission.csv
├── reports/         # Project report(s) (PDF/DOCX)
├── models/          # Trained checkpoints (not committed — see models/README.md)
├── requirements.txt
└── README.md
```

## Models

| # | Model | Type | MAP@3 (val) |
|---|-------|------|-------------|
| 0 | TF-IDF + Logistic Regression | Classical baseline | 0.9858 |
| 1 | BERT (bert-base-uncased) | Fine-tuned transformer | 0.9896 |
| 2 | BiLSTM + Attention | Custom deep learning model | 0.9988 |
| 3 | GRU | RNN-based model | *(see report)* |
| — | Weighted Ensemble | 0.15·TF-IDF + 0.25·BiLSTM + 0.60·BERT | 0.9988 |

Full details, architecture diagrams, and discussion are in `reports/`.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the competition dataset (`train.csv`, `test.csv`,
   `sample_submission.csv`) from Kaggle and place it in a local `data/`
   folder (or update `DATA_PATH` in `src/train.py` / `src/inference.py`).
3. Train all models:
   ```
   cd src
   python train.py
   ```
4. Generate the Kaggle submission file:
   ```
   python inference.py
   ```

## Git Workflow

This repository follows a milestone-based branching workflow:
- Each milestone is developed on its own branch (`milestone-1`, `milestone-2`, ...)
- Milestone branches are merged into `main` once complete
- Milestone branches are **never deleted**, for grading/audit purposes

## Kaggle

Original competition notebook: see `notebooks/`. Public leaderboard score: 0.74896.
