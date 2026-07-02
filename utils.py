"""
utils.py
Shared utilities for the Smart MCQ Solver project:
data loading, cleaning, PyTorch Dataset classes, and evaluation metrics.
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

SEED = 42
LABEL2ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
ID2LABEL = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
TEXT_COLUMNS = ["prompt", "A", "B", "C", "D", "E"]

MAX_VOCAB = 25000
MAX_LEN = 100


# ----------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean_data(data_path):
    """Load train/test/sample_submission CSVs and apply text cleaning + label encoding."""
    train = pd.read_csv(f"{data_path}/train.csv")
    test = pd.read_csv(f"{data_path}/test.csv")
    sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")

    for col in TEXT_COLUMNS:
        train[col] = train[col].apply(clean_text)
        test[col] = test[col].apply(clean_text)

    train["label"] = train["answer"].map(LABEL2ID)

    return train, test, sample_submission


# ----------------------------------------------------------------------
# Vocabulary building (for BiLSTM / GRU)
# ----------------------------------------------------------------------

def build_vocab(train_df, max_vocab=MAX_VOCAB):
    word_counter = Counter()
    for col in TEXT_COLUMNS:
        for text in train_df[col]:
            word_counter.update(text.lower().split())

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counter.most_common(max_vocab - 2):
        word2idx[word] = len(word2idx)

    return word2idx


def encode_text(text, word2idx, max_len=MAX_LEN):
    words = text.lower().split()
    ids = [word2idx.get(word, 1) for word in words]
    return ids[:max_len]


# ----------------------------------------------------------------------
# PyTorch Dataset + collate (shared by BiLSTM and GRU)
# ----------------------------------------------------------------------

class RNNDataset(Dataset):
    """Used for both the BiLSTM+Attention model and the plain GRU model."""

    def __init__(self, dataframe, word2idx, max_len=MAX_LEN):
        self.df = dataframe.reset_index(drop=True)
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        choices = []

        for option in ["A", "B", "C", "D", "E"]:
            text = row["prompt"] + " " + row[option]
            ids = encode_text(text, self.word2idx, self.max_len)
            choices.append(torch.tensor(ids, dtype=torch.long))

        return choices, int(row["label"])


def rnn_collate(batch):
    sequences = []
    labels = []

    for choices, label in batch:
        sequences.extend(choices)
        labels.append(label)

    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    sequences = sequences.view(len(batch), 5, -1)
    labels = torch.tensor(labels, dtype=torch.long)

    return sequences, labels


class BertMCQDataset(Dataset):
    """Dataset for the Hugging Face AutoModelForMultipleChoice pipeline (BERT)."""

    def __init__(self, dataframe, tokenizer, max_length=128, training=True):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        prompts = [row["prompt"]] * 5
        choices = [row["A"], row["B"], row["C"], row["D"], row["E"]]

        encoding = self.tokenizer(
            prompts,
            choices,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"]

        if self.training:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)

        return item


def make_pair_dataset(df):
    """Builds (prompt [SEP] option, is_correct) pairs for the TF-IDF baseline."""
    texts = []
    labels = []

    for _, row in df.iterrows():
        for option in ["A", "B", "C", "D", "E"]:
            text = row["prompt"] + " [SEP] " + row[option]
            texts.append(text)
            labels.append(1 if option == row["answer"] else 0)

    return texts, np.array(labels)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def map_at_3(y_true, probabilities):
    top3 = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    scores = []

    for actual, predicted in zip(y_true, top3):
        if actual == predicted[0]:
            scores.append(1.0)
        elif actual == predicted[1]:
            scores.append(0.5)
        elif actual == predicted[2]:
            scores.append(1 / 3)
        else:
            scores.append(0.0)

    return np.mean(scores)


def calculate_metrics(y_true, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average="macro")
    map3 = map_at_3(y_true, probabilities)
    return accuracy, macro_f1, map3


# ----------------------------------------------------------------------
# Model architectures
# ----------------------------------------------------------------------

class BiLSTMAttention(nn.Module):
    """Custom Deep Learning Model: BiLSTM + Attention, trained from scratch."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        batch_size, choices, seq_len = x.shape
        x = x.view(batch_size * choices, seq_len)

        mask = x != 0

        x = self.embedding(x)
        output, _ = self.lstm(x)

        attention_scores = self.attention(output).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        scores = self.classifier(context)

        return scores.view(batch_size, choices)


class GRUClassifier(nn.Module):
    """RNN-Based Model: plain unidirectional GRU, no attention."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, choices, seq_len = x.shape
        x = x.view(batch_size * choices, seq_len)

        lengths = (x != 0).sum(dim=1).clamp(min=1)

        x = self.embedding(x)
        output, _ = self.gru(x)

        last_hidden = output[torch.arange(output.size(0)), lengths - 1]
        scores = self.classifier(last_hidden)

        return scores.view(batch_size, choices)
