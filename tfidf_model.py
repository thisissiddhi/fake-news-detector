"""
tfidf_model.py
--------------
TF-IDF + Logistic Regression pipeline for fake news detection.
Runs on CPU — no GPU required.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from preprocess import load_isot_dataset, preprocess_for_tfidf, split_data

# Look for model in same directory as this script, or models/ subfolder
_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "tfidf_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_dir, "models", "tfidf_pipeline.pkl")


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
        )),
    ])


def train(true_path: str, fake_path: str) -> Pipeline:
    df = load_isot_dataset(true_path, fake_path)
    df = preprocess_for_tfidf(df)
    X_train, X_test, y_train, y_test = split_data(df, text_col="clean_text")

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    pipeline = build_pipeline()
    print("Training TF-IDF + Logistic Regression...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")

    return pipeline


def load_model() -> Pipeline:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(text: str, pipeline: Pipeline = None) -> dict:
    from preprocess import clean_text

    if pipeline is None:
        pipeline = load_model()

    cleaned = clean_text(text)
    proba = pipeline.predict_proba([cleaned])[0]
    label_idx = int(np.argmax(proba))
    label = "Real" if label_idx == 1 else "Fake"
    confidence = float(proba[label_idx])

    return {
        "label": label,
        "confidence": confidence,
        "fake_prob": float(proba[0]),
        "real_prob": float(proba[1]),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        train(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python tfidf_model.py <path/to/True.csv> <path/to/Fake.csv>")
