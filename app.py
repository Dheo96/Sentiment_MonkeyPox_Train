import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import clean_text

# Load model dan vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Konfigurasi tampilan
st.set_page_config(page_title="Monkeypox Sentiment Analyzer", layout="centered")
st.title("🧠 Monkeypox Tweet Sentiment Analyzer")
st.write("Masukkan tweet secara manual atau upload CSV untuk prediksi sentimen.")

# -------- Fitur 1: Input Manual --------
st.header("✍️ Ketik Tweet Manual")
user_input = st.text_input("Masukkan tweet:", "")

if user_input:
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    st.success(f"🔍 Prediksi Sentimen: **{pred.capitalize()}**")

# -------- Fitur 2: Upload CSV --------
st.header("📂 Upload CSV File")
uploaded_file = st.file_uploader("Unggah file CSV (kolom = Tweet_Text)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Tweet_Text' not in df.columns:
        st.error("❌ Kolom 'Tweet_Text' tidak ditemukan dalam file CSV.")
    else:
        st.write("📄 Contoh data:")
        st.dataframe(df[['Tweet_Text']].head())

        st.info("🔄 Memproses dan memprediksi tweet...")
        df['clean_text'] = df['Tweet_Text'].astype(str).apply(clean_text)
        X = vectorizer.transform(df['clean_text'])
        df['Predicted'] = model.predict(X)

        st.success("✅ Prediksi selesai!")

        # Visualisasi
        st.write("📊 Distribusi Sentimen:")
        sentiment_counts = df['Predicted'].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)
        ax.set_ylabel("Jumlah Tweet")
        ax.set_xlabel("Sentimen")
        st.pyplot(fig)

        # Download hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download hasil prediksi sebagai CSV",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )
