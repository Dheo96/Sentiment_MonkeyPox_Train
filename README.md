# 🧠 Monkeypox Tweet Sentiment Analyzer

A sentiment analysis tool for Monkeypox-related tweets using Logistic Regression and TF-IDF.
Supports:

## ✅ Requirements

Create and activate virtual environment (recommended):

```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run (Terminal CLI)

```bash
python app.py
```

You will be prompted to type tweets and get sentiment predictions directly in the terminal.

---

## 🧪 Notes
- Model is trained using 8000 labeled tweets
- TF-IDF uses unigram + bigram (n-gram 1–2)
- Preprocessing includes stemming and stopword removal using Sastrawi

---

## 📜 License
MIT — Free to use, modify, and share.

---

For BERT-based version or Streamlit dashboard, see future release or request via issue.
