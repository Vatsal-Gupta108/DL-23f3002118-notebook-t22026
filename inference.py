"""
inference.py
Loads the trained checkpoints (TF-IDF, BiLSTM, GRU, BERT), runs them on the
test set, combines predictions with the weighted ensemble, and writes submission.csv.

Run from the project root:
    python src/inference.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm.auto import tqdm

from utils import (
    ID2LABEL, load_and_clean_data, build_vocab,
    RNNDataset, rnn_collate, BertMCQDataset,
    BiLSTMAttention, GRUClassifier,
)

DATA_PATH = "../data"       # adjust to your local dataset path
MODEL_DIR = "../models"
OUTPUT_PATH = "../reports/submission.csv"

# Ensemble weights (adjust as needed once GRU validation results are in)
WEIGHTS = {"tfidf": 0.15, "bilstm": 0.20, "gru": 0.05, "bert": 0.60}


def predict_tfidf_test(tfidf, model, test_df):
    all_probs = []
    for _, row in test_df.iterrows():
        option_texts = [row["prompt"] + " [SEP] " + row[x] for x in ["A", "B", "C", "D", "E"]]
        features = tfidf.transform(option_texts)
        probs = model.predict_proba(features)[:, 1]
        probs = probs / probs.sum()
        all_probs.append(probs)
    return np.array(all_probs)


def predict_rnn_test(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="RNN test"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def predict_bert_test(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="BERT test"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train, test, sample_submission = load_and_clean_data(DATA_PATH)

    # --- TF-IDF ---
    # Note: TF-IDF vectorizer + LogisticRegression must be re-fit or loaded from
    # a saved pickle. If you saved them during training (e.g. with joblib),
    # load them here instead of re-fitting.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from utils import make_pair_dataset

    texts, labels = make_pair_dataset(train)
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(texts)
    tfidf_model = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    tfidf_model.fit(X_train, labels)
    tfidf_test_probs = predict_tfidf_test(tfidf, tfidf_model, test)

    # --- BiLSTM + GRU (shared vocab) ---
    word2idx = build_vocab(train)
    test_with_label = test.copy()
    test_with_label["label"] = 0  # placeholder, unused at inference

    rnn_test_loader = DataLoader(
        RNNDataset(test_with_label, word2idx), batch_size=64, shuffle=False, collate_fn=rnn_collate
    )

    bilstm_model = BiLSTMAttention(len(word2idx)).to(device)
    bilstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/best_bilstm.pt", map_location=device))
    bilstm_test_probs = predict_rnn_test(bilstm_model, rnn_test_loader, device)

    gru_model = GRUClassifier(len(word2idx)).to(device)
    gru_model.load_state_dict(torch.load(f"{MODEL_DIR}/best_gru.pt", map_location=device))
    gru_test_probs = predict_rnn_test(gru_model, rnn_test_loader, device)

    # --- BERT ---
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_test_loader = DataLoader(
        BertMCQDataset(test, tokenizer, training=False), batch_size=16, shuffle=False
    )

    bert_model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased").to(device)
    bert_model.load_state_dict(torch.load(f"{MODEL_DIR}/best_bert.pt", map_location=device))
    bert_test_probs = predict_bert_test(bert_model, bert_test_loader, device)

    # --- Weighted ensemble ---
    final_probs = (
        WEIGHTS["tfidf"] * tfidf_test_probs
        + WEIGHTS["bilstm"] * bilstm_test_probs
        + WEIGHTS["gru"] * gru_test_probs
        + WEIGHTS["bert"] * bert_test_probs
    )

    top3_indices = np.argsort(final_probs, axis=1)[:, ::-1][:, :3]

    predictions = []
    for row in top3_indices:
        letters = [ID2LABEL[int(idx)] for idx in row]
        predictions.append(" ".join(letters))

    submission = sample_submission.copy()
    submission["Prediction"] = predictions
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"submission.csv created successfully at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
