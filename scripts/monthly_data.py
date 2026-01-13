import requests
import pandas as pd
import sys

API_KEY = ""YOUR_API_KEY""
SYMBOL = "AMGN"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={SYMBOL}&apikey={API_KEY}"

r = requests.get(url)
data = r.json()


df = pd.DataFrame.from_dict(data["Monthly Time Series"], orient="index")

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

df.to_csv("AMGN_monthly.csv")

