"""
train.py
Trains all four models for the Smart MCQ Solver project:
1. TF-IDF + Logistic Regression (baseline)
2. BiLSTM + Attention (custom deep learning model)
3. GRU (plain RNN-based model)
4. BERT (fine-tuned transformer, bert-base-uncased)

Run from the project root:
    python src/train.py
"""

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm.auto import tqdm
import wandb

from utils import (
    SEED, ID2LABEL,
    load_and_clean_data, build_vocab,
    RNNDataset, rnn_collate, BertMCQDataset, make_pair_dataset,
    calculate_metrics,
    BiLSTMAttention, GRUClassifier,
)

DATA_PATH = "../data"  # adjust to your local dataset path
MODEL_DIR = "../models"
WANDB_PROJECT = "smart-mcq-solver"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# 1. TF-IDF + Logistic Regression
# ----------------------------------------------------------------------

def train_tfidf(train_df, val_df):
    texts, labels = make_pair_dataset(train_df)

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(texts)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    model.fit(X_train, labels)

    def predict(df):
        all_probs = []
        for _, row in df.iterrows():
            option_texts = [row["prompt"] + " [SEP] " + row[x] for x in ["A", "B", "C", "D", "E"]]
            features = tfidf.transform(option_texts)
            probs = model.predict_proba(features)[:, 1]
            probs = probs / probs.sum()
            all_probs.append(probs)
        return np.array(all_probs)

    val_probs = predict(val_df)
    acc, f1, map3 = calculate_metrics(val_df["label"].values, val_probs)

    wandb.init(project=WANDB_PROJECT, name="01_TFIDF_LogisticRegression")
    wandb.log({"accuracy": acc, "macro_f1": f1, "map3": map3})
    wandb.finish()

    print(f"TF-IDF   | Accuracy {acc:.4f} | F1 {f1:.4f} | MAP@3 {map3:.4f}")
    return tfidf, model, predict, val_probs


# ----------------------------------------------------------------------
# 2 & 3. RNN-family models (BiLSTM+Attention, GRU) share a training loop
# ----------------------------------------------------------------------

def train_rnn_model(model, model_name, run_name, checkpoint_path,
                     train_loader, val_loader, device, epochs, lr=0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    wandb.init(project=WANDB_PROJECT, name=run_name)
    best_map3 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_acc, val_f1, val_map3, val_probs = evaluate_rnn_model(model, val_loader, device)

        print(f"{model_name} Epoch {epoch+1} | Loss {train_loss:.4f} | "
              f"Accuracy {val_acc:.4f} | F1 {val_f1:.4f} | MAP@3 {val_map3:.4f}")

        wandb.log({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_accuracy": val_acc, "val_macro_f1": val_f1, "val_map3": val_map3,
        })

        if val_map3 > best_map3:
            best_map3 = val_map3
            torch.save(model.state_dict(), checkpoint_path)

    wandb.finish()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return evaluate_rnn_model(model, val_loader, device)


def evaluate_rnn_model(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    acc, f1, map3 = calculate_metrics(all_labels, all_probs)
    return acc, f1, map3, all_probs


# ----------------------------------------------------------------------
# 4. BERT (fine-tuned transformer)
# ----------------------------------------------------------------------

def train_bert(train_df, val_df, device, epochs=2, checkpoint_path=f"{MODEL_DIR}/best_bert.pt"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(BertMCQDataset(train_df, tokenizer), batch_size=8, shuffle=True)
    val_loader = DataLoader(BertMCQDataset(val_df, tokenizer), batch_size=16, shuffle=False)

    model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased").to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    wandb.init(project=WANDB_PROJECT, name="03_BERT_MultipleChoice")
    best_map3 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"BERT Epoch {epoch+1}"):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        acc, f1, map3, _ = evaluate_bert(model, val_loader, device)

        print(f"BERT Epoch {epoch+1} | Loss {train_loss:.4f} | "
              f"Accuracy {acc:.4f} | F1 {f1:.4f} | MAP@3 {map3:.4f}")

        wandb.log({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_accuracy": acc, "val_macro_f1": f1, "val_map3": map3,
        })

        if map3 > best_map3:
            best_map3 = map3
            torch.save(model.state_dict(), checkpoint_path)

    wandb.finish()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return (tokenizer, model, *evaluate_bert(model, val_loader, device))


def evaluate_bert(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="BERT validation"):
            all_labels.extend(batch["labels"].numpy())
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    acc, f1, map3 = calculate_metrics(all_labels, all_probs)
    return acc, f1, map3, all_probs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train, test, sample_submission = load_and_clean_data(DATA_PATH)
    train_df, val_df = train_test_split(
        train, test_size=0.20, random_state=SEED, stratify=train["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 1. TF-IDF baseline
    tfidf, tfidf_model, tfidf_predict, tfidf_val_probs = train_tfidf(train_df, val_df)

    # Shared vocab + loaders for BiLSTM and GRU
    word2idx = build_vocab(train_df)

    train_loader = DataLoader(
        RNNDataset(train_df, word2idx), batch_size=32, shuffle=True, collate_fn=rnn_collate
    )
    val_loader = DataLoader(
        RNNDataset(val_df, word2idx), batch_size=64, shuffle=False, collate_fn=rnn_collate
    )

    # 2. BiLSTM + Attention (custom DL model)
    bilstm_model = BiLSTMAttention(len(word2idx)).to(device)
    bilstm_acc, bilstm_f1, bilstm_map3, bilstm_val_probs = train_rnn_model(
        bilstm_model, "BiLSTM", "02_BiLSTM_Attention", f"{MODEL_DIR}/best_bilstm.pt",
        train_loader, val_loader, device, epochs=8,
    )
    print(f"BiLSTM   | Accuracy {bilstm_acc:.4f} | F1 {bilstm_f1:.4f} | MAP@3 {bilstm_map3:.4f}")

    # 3. GRU (RNN-based model)
    gru_model = GRUClassifier(len(word2idx)).to(device)
    gru_acc, gru_f1, gru_map3, gru_val_probs = train_rnn_model(
        gru_model, "GRU", "05_GRU_RNN", f"{MODEL_DIR}/best_gru.pt",
        train_loader, val_loader, device, epochs=8,
    )
    print(f"GRU      | Accuracy {gru_acc:.4f} | F1 {gru_f1:.4f} | MAP@3 {gru_map3:.4f}")

    # 4. BERT (fine-tuned transformer)
    tokenizer, bert_model, bert_acc, bert_f1, bert_map3, bert_val_probs = train_bert(
        train_df, val_df, device, epochs=2,
    )
    print(f"BERT     | Accuracy {bert_acc:.4f} | F1 {bert_f1:.4f} | MAP@3 {bert_map3:.4f}")

    print("\nAll models trained. Checkpoints saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
