import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# LOAD DATA
train = pd.read_csv("train.csv", index_col=0, parse_dates=True)
val = pd.read_csv("validation.csv", index_col=0, parse_dates=True)

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_val = val.drop(columns=["close", "Year", "Month"])
y_val = val["close"]

# CFA - CORRELATION ANALYSIS
corr = X_train.join(y_train).corr()["close"]
corr_sorted = corr.sort_values(key=abs, ascending=False)

THRESHOLD = 0.3

selected = corr_sorted[abs(corr_sorted) > THRESHOLD].index.tolist()
if "close" in selected:
    selected.remove("close")

# PRINT CORRELATIONS
print("\nCFA CORRELATION WITH TARGET (sorted by |corr|):")
print(corr_sorted.to_string())

print("\nThreshold:", THRESHOLD)
print("Selected features:")
for f in selected:
    print(" -", f)

print("Total selected features:", len(selected))

# MODEL TRAINING
model = LinearRegression()
model.fit(X_train[selected], y_train)


# PREDICTIONS
train_pred = model.predict(X_train[selected])
val_pred = model.predict(X_val[selected])

# METRICS FUNCTION
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    return mae, rmse

# PRINT METRICS
print("\nCFA MODEL RESULTS")
train_mae, train_rmse = evaluate("TRAIN SET", y_train, train_pred)
val_mae, val_rmse = evaluate("VALIDATION SET", y_val, val_pred)

# PRINT MODEL COEFFICIENTS
coef_df = pd.DataFrame({
    "Feature": selected,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nCFA MODEL COEFFICIENTS (sorted by importance):")
print(coef_df.to_string(index=False))
