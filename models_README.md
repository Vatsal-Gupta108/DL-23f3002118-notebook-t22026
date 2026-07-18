# Models

Trained model checkpoints are not committed to this repository because they
are large binary files (BERT and BiLSTM/GRU checkpoints can be 100MB+),
and GitHub is not designed to store large binaries efficiently.

## Checkpoints produced by `src/train.py`

Running `src/train.py` will save the following files into this folder:

- `best_bilstm.pt`   — best BiLSTM + Attention checkpoint (by validation MAP@3)
- `best_gru.pt`      — best GRU checkpoint (by validation MAP@3)
- `best_bert.pt`     — best fine-tuned BERT checkpoint (by validation MAP@3)

The TF-IDF + Logistic Regression model is not saved to disk in the current
scripts; it is re-fit at inference time inside `src/inference.py`. If you
want to persist it, save it with `joblib.dump()` after training.

## How to get the checkpoints

Since the checkpoints are not in this repo, to reproduce results:

1. Run `src/train.py` (or the original Kaggle notebook in `notebooks/`) to
   train all models from scratch — this will regenerate the checkpoints here.
2. Alternatively, if checkpoints were shared separately (e.g. via Google Drive
   or Kaggle notebook output), download them and place them directly in this
   folder using the exact filenames above before running `src/inference.py`.

## Note on Kaggle

The original notebook was run and trained on Kaggle, where GPU checkpoints
were saved to `/kaggle/working/`. This folder mirrors that same set of
checkpoint filenames for local/offline use.
