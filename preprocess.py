"""
preprocess.py
-------------
Text preprocessing pipeline using NLTK.
Handles stopword removal, lemmatization, and sequence padding for BERT.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download required NLTK resources
def download_nltk_resources():
    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Lowercase, remove special characters, stopwords, and optionally lemmatize.
    """
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def load_isot_dataset(true_path: str, fake_path: str) -> pd.DataFrame:
    """
    Load the ISOT dataset (True.csv and Fake.csv) and return a combined DataFrame.
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = 1  # Real
    fake_df["label"] = 0  # Fake

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[["title", "text", "label"]].copy()
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df.dropna(subset=["content"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def preprocess_for_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full text cleaning pipeline for TF-IDF model."""
    print("Cleaning text for TF-IDF (this may take a minute)...")
    df = df.copy()
    df["clean_text"] = df["content"].apply(clean_text)
    return df


def split_data(df: pd.DataFrame, text_col: str, label_col: str = "label", test_size: float = 0.2):
    """Split into train/test sets."""
    X = df[text_col]
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


if __name__ == "__main__":
    # Quick sanity check
    sample = "Breaking: Scientists discover shocking truth about vaccines!! Click here!!"
    print("Raw:", sample)
    print("Cleaned:", clean_text(sample))
