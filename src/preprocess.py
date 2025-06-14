import re
import pandas as pd
import nltk
from tqdm import tqdm

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download tokenizer
nltk.download('punkt', quiet=True)

stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

tqdm.pandas()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return " ".join(tokens)

def preprocess_dataframe(path):
    print("📂 Membaca seluruh dataset...")
    df = pd.read_csv(path)

    # Rename agar sesuai pipeline
    df = df.rename(columns={'cleaned_tweet': 'Tweet_Text', 'sentiment': 'Sentiment'})
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

    print(f"📊 Jumlah tweet yang diproses: {len(df)}")
    print("🧹 Preprocessing dengan stemming + stopword...")

    df['clean_text'] = df['Tweet_Text'].astype(str).progress_apply(clean_text)

    return df
