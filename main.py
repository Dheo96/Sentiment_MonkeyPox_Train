from src.preprocess import preprocess_dataframe, clean_text
from src.train_model import train_model
from src.evaluate_model import evaluate
import pickle

# Coba import interpretasi model (LIME/SHAP)
try:
    from src.explain_model import explain_with_lime, explain_with_shap
    has_explain = True
except ImportError:
    print("⚠️ Modul LIME/SHAP tidak ditemukan. Lewati interpretasi model.")
    has_explain = False

def predict_single_text(text):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    print(f"\n🧾 Kalimat: \"{text}\"\n🔍 Prediksi Emosi: {pred}")

def main():
    print("📦 Preprocessing & load dataset...")
    df = preprocess_dataframe("data/monkeypox_sentiment_output.csv")

    print("✅ Training model...")
    model, tfidf, X_test, y_test = train_model(df)

    # Simpan model dan TF-IDF vectorizer (ditimpa jika sudah ada)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("💾 Model & vectorizer disimpan sebagai model.pkl dan vectorizer.pkl")

    print("📊 Evaluasi model:")
    evaluate(model, X_test, y_test)

    if has_explain:
        try:
            print("🧠 Interpretasi model (SHAP + LIME)...")
            explain_with_shap(model, tfidf, df['clean_text'].tolist())
            explain_with_lime(model, tfidf, model.classes_, df['clean_text'].iloc[0])
        except Exception as e:
            print(f"❌ Gagal interpretasi model: {e}")

    # Simpan hasil prediksi seluruh tweet
    print("\n💾 Menyimpan hasil prediksi semua tweet ke 'output.csv'...")
    X_all = tfidf.transform(df['clean_text'])
    y_pred_all = model.predict(X_all)

    df_output = df.copy()
    df_output['Predicted'] = y_pred_all
    df_output.to_csv("output.csv", index=False)
    print("✅ File 'output.csv' berhasil dibuat.")

    # Input interaktif
    print("\n💬 Masukkan tweet untuk diprediksi (ketik 'exit' untuk keluar).")
    while True:
        user_input = input("\nKetik tweet: ")
        if user_input.lower() == "exit":
            print("👋 Terima kasih, program selesai.")
            break
        predict_single_text(user_input)

if __name__ == "__main__":
    main()
