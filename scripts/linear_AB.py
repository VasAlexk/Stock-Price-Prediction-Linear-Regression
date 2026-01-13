import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# data
df = pd.read_csv("AMGN_monthly.csv", index_col=0, parse_dates=True)
df.sort_index(inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month

# lag features
N = 3
for i in range(1, N + 1):
    df[f"close_t-{i}"] = df["close"].shift(i)
    df[f"volume_t-{i}"] = df["volume"].shift(i)

df.dropna(inplace=True)

# split ta data
train = df[df["Year"] < 2024]
test = df[df["Year"] >= 2024]

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_test = test.drop(columns=["close", "Year", "Month"])
y_test = test["close"]

# model
model = LinearRegression()
model.fit(X_train, y_train)

# method a
pred_A = model.predict(X_test)
mae_A = mean_absolute_error(y_test, pred_A)

# method b 
history = X_test.iloc[0:1].copy()

actuals_B, preds_B = [], []

for i in range(len(X_test)):

    X = history[X_train.columns]
    pred = model.predict(X)[0]

    preds_B.append(pred)
    actuals_B.append(y_test.iloc[i])

    new = history.copy()

    for j in reversed(range(2, N + 1)):
        new[f"close_t-{j}"] = new[f"close_t-{j-1}"]

    new["close_t-1"] = pred

    # pagwnw ta ypoloipa
    history = new.copy()

# metrikes
mae_B = mean_absolute_error(actuals_B, preds_B)

# horizon mae
mae_horizon = []
for k in range(1, len(actuals_B) + 1):
    mae_horizon.append(mean_absolute_error(actuals_B[:k], preds_B[:k]))

# apotelesmata
print("\nLINEAR MODEL RESULTS")
print("METHOD A MAE:", round(mae_A, 3))
print("METHOD B MAE:", round(mae_B, 3))

THRESHOLD = 10
valid = [i + 1 for i, m in enumerate(mae_horizon) if m <= THRESHOLD]

if valid:
    print(f"Reliable horizon (MAE ≤ {THRESHOLD}): {max(valid)} months")
else:
    print("No reliable horizon under threshold")

# csv save
pd.DataFrame({
    "actual": y_test.values,
    "predicted": pred_A
}).to_csv("linear_method_A.csv", index=False)

pd.DataFrame({
    "horizon": range(1, len(preds_B) + 1),
    "actual": actuals_B,
    "predicted": preds_B,
    "abs_error": np.abs(np.array(actuals_B) - np.array(preds_B))
}).to_csv("linear_method_B.csv", index=False)

pd.DataFrame({
    "horizon": range(1, len(mae_horizon) + 1),
    "mae": mae_horizon
}).to_csv("linear_horizon.csv", index=False)
