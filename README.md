# 📰 Fake News Detector

A machine learning pipeline for detecting fake vs. real news articles using **TF-IDF + Logistic Regression** and a fine-tuned **DistilBERT** model, with a real-time **Gradio** web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

---

## 🧠 Models

| Model | Accuracy | Speed | Hardware |
|-------|----------|-------|----------|
| TF-IDF + Logistic Regression | ~98% (ISOT) | Fast | CPU |
| DistilBERT (fine-tuned) | ~96% | Slower | GPU recommended |

---

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   └── README.md              # Dataset download instructions
├── src/
│   ├── preprocess.py          # NLTK text cleaning pipeline
│   ├── tfidf_model.py         # TF-IDF + Logistic Regression
│   ├── bert_model.py          # DistilBERT fine-tuning & inference
│   └── app.py                 # Gradio web interface
├── notebooks/
│   └── train_distilbert.ipynb # Google Colab training notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

See [`data/README.md`](data/README.md) for instructions. Place `True.csv` and `Fake.csv` in the `data/` folder.

---

## 🔧 Training

### TF-IDF Model (CPU, ~2 min)

```bash
cd src
python tfidf_model.py ../data/True.csv ../data/Fake.csv
```

### DistilBERT Model (GPU required, ~25 min on Colab)

Open the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fake-news-detector/blob/main/notebooks/train_distilbert.ipynb)

Then download the trained model and place it in `models/distilbert/`.

---

## 🌐 Running the App

```bash
cd src
python app.py
```

Open your browser at `http://localhost:7860`

---

## 📊 Dataset

**ISOT Fake News Dataset** from the University of Victoria:
- ~21,417 real news articles (Reuters)
- ~23,481 fake news articles
- Total: ~44,898 articles

Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 🛠 Tech Stack

- **NLP**: NLTK, HuggingFace Transformers
- **ML**: scikit-learn, PyTorch
- **Models**: TF-IDF, DistilBERT (`distilbert-base-uncased`)
- **UI**: Gradio
- **Training**: Google Colab (free GPU)

---

## 📄 License

MIT
