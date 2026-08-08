# 📰 AI-Powered Fake News Detection Using Text Classification

A machine learning project that classifies news articles as **Real** or **Fake** using classical NLP techniques (TF-IDF) and multiple supervised classification algorithms — achieving up to **99.55% accuracy**.

---

## 📌 Overview

The rapid spread of misinformation on digital platforms makes manual fact-checking impractical at scale. This project builds a text-classification pipeline that automatically flags a news article as genuine or fabricated based purely on its text content.

Four classifiers are trained and benchmarked on the same TF-IDF feature space:

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 98.99% | 0.99 | 0.99 | 0.99 |
| Decision Tree Classifier | **99.55%** | 1.00 | 1.00 | 1.00 |
| Random Forest Classifier | 99.03% | 0.99 | 0.99 | 0.99 |
| Gradient Boosting Classifier | 99.54% | 1.00 | 1.00 | 1.00 |

*(macro-averaged metrics, evaluated on a 13,470-article held-out test set)*

---

## 🗂️ Dataset

- **Source:** [Fake and Real News Dataset](https://www.kaggle.com/) — `True.csv` (real, Reuters-sourced) and `Fake.csv` (fake/unreliable) articles.
- **Size:** 44,898 total articles after merging (31,428 train / 13,470 test, 70/30 split).
- **Columns used:** `text` (article body) and `label` (1 = Real, 0 = Fake).

> **Note:** The dataset CSVs are not included in this repository due to size/licensing. Download `True.csv` and `Fake.csv` from Kaggle and place them in the project root before running the notebook.

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colaboratory / Jupyter Notebook
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `re`

---

## 🧠 Pipeline

```
Raw CSV Data
   │
   ▼
Labelling & Merging (True.csv=1, Fake.csv=0)
   │
   ▼
Data Cleaning (drop nulls, drop irrelevant columns, shuffle)
   │
   ▼
Text Preprocessing (wordopt: lowercase, remove URLs/punctuation/HTML/digits/newlines)
   │
   ▼
Train/Test Split (70/30)
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Model Training (Logistic Regression | Decision Tree | Random Forest | Gradient Boosting)
   │
   ▼
Evaluation (Accuracy, Precision, Recall, F1-score)
   │
   ▼
Manual Testing Utility (classify any new article)
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Clone the repository
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### Add the dataset
Download `True.csv` and `Fake.csv` from Kaggle and place them in the project root.

### Run
Open `FakeNewsDetection.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run all cells in order.

---

## 🧪 Usage — Manual Testing

Once the models are trained, classify any new article:

```python
news_article = str(input())
manual_testing(news_article)
```

**Example output:**
```
LR Prediction: It is Fake News
GBC Prediction: It is Fake News
RFC Prediction: It is Fake News
```

---

## 📁 Project Structure

```
fake-news-detection/
├── FakeNewsDetection.ipynb   # Main notebook (full pipeline)
├── True.csv                  # Real news data (not included — download separately)
├── Fake.csv                  # Fake news data (not included — download separately)
├── README.md
└── requirements.txt
```

---

## 📦 requirements.txt

```
pandas
numpy
scikit-learn
```

---

## 🔮 Future Scope

- Incorporate deep learning / transformer-based models (LSTM, BERT) for improved generalization.
- Deploy as a web app using Flask or Streamlit for real-time article verification.
- Add explainability (SHAP/LIME) to surface which words drive each prediction.
- Test robustness against paraphrased or adversarial fake articles.

---

## 👤 Author

**Devanshu Gaidhane**
B.Tech, Computer Science & Engineering
MIT Art, Design and Technology University, Pune

---

## 📄 License

This project is released under the [MIT License](LICENSE).
