import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

    # Save
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

    return model, tfidf, X_test, y_test
