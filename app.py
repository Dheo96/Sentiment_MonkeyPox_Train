import pickle
from src.preprocess import clean_text

# Load model dan vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("🧠 Monkeypox Sentiment Analyzer (Terminal Version)")
print("Ketik tweet untuk diprediksi (atau ketik 'exit' untuk keluar)")

while True:
    user_input = input("\nTweet: ")
    if user_input.lower() == "exit":
        print("👋 Terima kasih! Program selesai.")
        break

    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    print(f"🔍 Prediksi Sentimen: {pred.capitalize()}")
