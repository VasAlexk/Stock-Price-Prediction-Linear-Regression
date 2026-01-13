import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# data
df = pd.read_csv("AMGN_monthly.csv", index_col=0, parse_dates=True)
df.sort_index(inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month

# lags
N = 3
for i in range(1, N+1):
    df[f"close_t-{i}"] = df["close"].shift(i)
    df[f"volume_t-{i}"] = df["volume"].shift(i)

df.dropna(inplace=True)


# selected features apo CFA
selected = [
    "close_t-1", "high", "low", "open",
    "close_t-2", "close_t-3",
    "volume_t-3", "volume_t-2", "volume_t-1"
]

# vazw volume  
selected.append("volume")

# teliko training 
X = df[selected]
y = df["close"]

model = LinearRegression()
model.fit(X, y)

# recursive forecast
last = df.iloc[-1:].copy()
predictions = []

future_dates = [
    pd.Timestamp("2025-12-01"),
    pd.Timestamp("2026-01-01")
]

for date in future_dates:

    X_last = last[selected]
    pred = model.predict(X_last)[0]
    predictions.append(pred)

    # shift ta lags
    new = last.copy()
    new["close"] = pred

    for i in range(N, 1, -1):
        new[f"close_t-{i}"] = new[f"close_t-{i-1}"]
    new["close_t-1"] = pred

    # volume kai OHLC opws einai
    last = new.copy()

# apotelesmata
print("\nRESULTS")
for d, p in zip(future_dates, predictions):
    print(d.strftime("%B %Y"), ":", round(p, 2))
