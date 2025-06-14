import pandas as pd

df = pd.read_csv("data/monkeypox_sentiment_output.csv")
print("Kolom tersedia:")
print(df.columns)
print("\nContoh 5 data pertama:")
print(df.head())
