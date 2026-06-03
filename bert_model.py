"""
bert_model.py
-------------
DistilBERT fine-tuning for fake news detection.
Designed to run on GPU (Google Colab recommended).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SAVE_DIR = "models/distilbert"


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(true_path: str, fake_path: str):
    """Fine-tune DistilBERT on the ISOT dataset."""
    from preprocess import load_isot_dataset, split_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_isot_dataset(true_path, fake_path)
    # Use raw text (no lemmatization) — BERT handles its own tokenization
    df["content"] = df["content"].str[:1024]  # Truncate very long texts early

    X_train, X_test, y_train, y_test = split_data(df, text_col="content")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} — Avg loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

    # Save model and tokenizer
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}/")


def load_model(model_dir: str = SAVE_DIR):
    """Load fine-tuned DistilBERT model and tokenizer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict(text: str, model=None, tokenizer=None) -> dict:
    """Predict real/fake for a single article."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )
        proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    label_idx = int(np.argmax(proba))
    return {
        "label": "Real" if label_idx == 1 else "Fake",
        "confidence": float(proba[label_idx]),
        "fake_prob": float(proba[0]),
        "real_prob": float(proba[1]),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        train(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python bert_model.py <path/to/True.csv> <path/to/Fake.csv>")
