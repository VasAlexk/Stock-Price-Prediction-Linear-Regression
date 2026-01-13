
import requests
import pandas as pd
import sys

API_KEY = ""YOUR_API_KEY""
SYMBOL = "AMGN"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}&outputsize=compact"

response = requests.get(url)
data = response.json()


# Μετατροπη σε dataframe
df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")

df.rename(columns={
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume"
}, inplace=True)

df = df.astype(float)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

filename = "AMGN.csv"
df.to_csv(filename)

