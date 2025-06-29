# 📱 WhatsApp Chat Analyzer

A powerful Streamlit-based web app that processes WhatsApp chat exports and provides insightful visualizations, sentiment analysis, emoji stats, word clouds, and machine learning–based spam detection.

---

## 🚀 Features

- 📊 **User Statistics**: Total messages, word count, media, and shared links.
- 📅 **Monthly Timeline Forecast**: Actual vs predicted message trends using Polynomial Regression.
- 😄 **Sentiment Analysis**: Positive / Negative / Neutral classification using VADER.
- ☁️ **Word Cloud**: Highlights most frequent words excluding stopwords.
- 😎 **Emoji Analysis**: Shows emoji usage by count and percentage.
- 🛡️ **Spam Detection**:
  - 🔍 Uses both **Random Forest** and **XGBoost** classifiers.
  - ✅ Visualized with confusion matrix and accuracy scores.
- 📈 **Interactive Visuals**: Built with `matplotlib`, `seaborn`, and `Streamlit`.

---

## 📂 How to Use

1. Export your WhatsApp chat as `.txt` (without media).
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
