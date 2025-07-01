import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import preprocesser
import helper

nltk.download('vader_lexicon')

# Sidebar
st.sidebar.title("\U0001F4F1 WhatsApp Chat Processor")

uploaded_file = st.sidebar.file_uploader("\U0001F4C2 Upload a WhatsApp chat file", type=['txt'])

if uploaded_file is not None:
    with st.spinner("Processing chat..."):
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocesser.preprocess(data)

    st.header("\U0001F9FE Processed Chat Data")
    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "overall")
    selected_user = st.sidebar.selectbox("\U0001F4CA Show Analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        st.title("\U0001F4C8 Top Statistics")
        num_messages, words, num_mediaquery, links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Messages")
            st.write(num_messages)
        with col2:
            st.header("Words")
            st.write(words)
        with col3:
            st.header("Media")
            st.write(num_mediaquery)
        with col4:
            st.header("Links")
            st.write(links)

        # Monthly Timeline + Forecast
        st.title('\U0001F5D3️ Monthly Timeline with Forecast')
        timeline = helper.monthly_timeline(selected_user, df)
        timeline['time_index'] = np.arange(len(timeline))

        X = timeline[['time_index']]
        y = timeline['message']

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        future_indices = np.arange(len(timeline) + 2).reshape(-1, 1)
        future_poly = poly.transform(future_indices)
        predicted = model.predict(future_poly)

        future_labels = timeline['time'].tolist() + ['June-2025', 'July-2025']

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_labels, predicted, color='blue', marker='o', label='Predicted Trend', linewidth=2.5)
        ax.scatter(timeline['time'], timeline['message'], color='green', s=100, label='Actual')

        for i, val in enumerate(timeline['message']):
            ax.text(i, val + 30, str(val), ha='center', fontsize=9)

        ax.set_title('Monthly Message Trend with Forecast', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Message Count')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # Sentiment Analysis
        st.title("\U0001F60A Sentiment Analysis")
        sia = SentimentIntensityAnalyzer()
        df['sentiment'] = df['message'].apply(lambda msg: sia.polarity_scores(msg)['compound'])
        df['sentiment_label'] = df['sentiment'].apply(
            lambda s: 'Positive' if s > 0 else ('Negative' if s < 0 else 'Neutral'))

        sentiment_counts = df['sentiment_label'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        st.pyplot(fig)

        # Word Cloud
        st.title("\u2601\ufe0f Word Cloud")
        text = " ".join(msg for msg in df['message'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Emoji Analysis
        st.title("\U0001F60E Emoji Analysis")
        more_emojis = helper.emoji_helper(selected_user, df)
        if not more_emojis.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(more_emojis)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(more_emojis['Count'], labels=more_emojis['Emoji'], autopct='%1.1f%%')
                st.pyplot(fig)

        # Spam Detection
        st.title("\U0001F6E1️ Spam Detection (4 Classifiers)")
        df['spam'] = np.random.choice([0, 1], size=len(df))  # Dummy labels
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X = vectorizer.fit_transform(df['message'].fillna(""))
        X_train, X_test, y_train, y_test = train_test_split(X, df['spam'], test_size=0.2, random_state=42)

        # Logistic Regression
        st.subheader("📊 Logistic Regression")
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)
        log_pred = log_model.predict(X_test)
        log_acc = accuracy_score(y_test, log_pred)
        st.write(f"Accuracy: **{log_acc * 100:.2f}%**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, log_pred, ax=ax)
        st.pyplot(fig)
        st.text(classification_report(y_test, log_pred))

        # Naive Bayes
        st.subheader("🤖 Naive Bayes")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)
        st.write(f"Accuracy: **{nb_acc * 100:.2f}%**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, nb_pred, ax=ax)
        st.pyplot(fig)
        st.text(classification_report(y_test, nb_pred))

        # SVM
        st.subheader("🧠 Support Vector Machine")
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        st.write(f"Accuracy: **{svm_acc * 100:.2f}%**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, svm_pred, ax=ax)
        st.pyplot(fig)
        st.text(classification_report(y_test, svm_pred))

        # Random Forest
        st.subheader("🌲 Random Forest")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        st.write(f"Accuracy: **{rf_acc * 100:.2f}%**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=ax)
        st.pyplot(fig)
        st.text(classification_report(y_test, rf_pred))

        # Accuracy Comparison Graph
        st.subheader("📊 Accuracy Comparison Chart")
        model_names = ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"]
        accuracies = [log_acc * 100, nb_acc * 100, svm_acc * 100, rf_acc * 100]

        fig, ax = plt.subplots()
        bars = ax.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'purple'])
        ax.set_ylim([0, 100])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom')

        st.pyplot(fig)

else:
    st.sidebar.info("\U0001F4CE Please upload a WhatsApp chat file to begin.")
