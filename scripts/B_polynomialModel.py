import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# fortwsh data
df = pd.read_csv("AMGN_monthly.csv", index_col=0, parse_dates=True)
df.sort_index(inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month

# dhmioyrgia lags
N = 3
for i in range(1, N + 1):
    df[f"close_t-{i}"] = df["close"].shift(i)
    df[f"volume_t-{i}"] = df["volume"].shift(i)

df.dropna(inplace=True)


#  split data gia train/val

train = df[df["Year"] < 2024]
test = df[df["Year"] >= 2024]

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_test = test.drop(columns=["close", "Year", "Month"])
y_test = test["close"]


# polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\nFeature Count")
print("Original features:", X_train.shape[1])
print("Polynomial features:", X_train_poly.shape[1])


#  scaling
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)


#  models
lasso = Lasso(alpha=0.05, max_iter=10000)
ridge = Ridge(alpha=10)

lasso.fit(X_train_poly, y_train)
ridge.fit(X_train_poly, y_train)


#  Method A
pred_A_lasso = lasso.predict(X_test_poly)
pred_A_ridge = ridge.predict(X_test_poly)

mae_A_lasso = mean_absolute_error(y_test, pred_A_lasso)
mae_A_ridge = mean_absolute_error(y_test, pred_A_ridge)

rmse_A_lasso = np.sqrt(mean_squared_error(y_test, pred_A_lasso))
rmse_A_ridge = np.sqrt(mean_squared_error(y_test, pred_A_ridge))

#  Method B recursive forecasting
def recursive_forecast(model, history, steps):
    actuals, preds = [], []
    history = history.copy()

    for i in range(steps):
        X_hist = poly.transform(history[X_train.columns])
        X_hist = scaler.transform(X_hist)

        pred = model.predict(X_hist)[0]
        preds.append(pred)
        actuals.append(test.iloc[i]["close"])

        new = history.copy()
        new["close"] = pred

        for j in reversed(range(2, N + 1)):
            new[f"close_t-{j}"] = new[f"close_t-{j-1}"]

        new["close_t-1"] = pred
        history = new.copy()

    return actuals, preds


history_start = test.iloc[0:1].copy()

actual_lasso, pred_lasso = recursive_forecast(lasso, history_start, len(test))
actual_ridge, pred_ridge = recursive_forecast(ridge, history_start, len(test))

mae_B_lasso = mean_absolute_error(actual_lasso, pred_lasso)
mae_B_ridge = mean_absolute_error(actual_ridge, pred_ridge)

rmse_B_lasso = np.sqrt(mean_squared_error(actual_lasso, pred_lasso))
rmse_B_ridge = np.sqrt(mean_squared_error(actual_ridge, pred_ridge))

# output 
print("\nPOLYNOMIAL MODEL RESULTS")

print("\nMETHOD A")
print("LASSO MAE:", round(mae_A_lasso, 2))
print("RIDGE MAE:", round(mae_A_ridge, 2))
print("LASSO RMSE:", round(rmse_A_lasso, 2))
print("RIDGE RMSE:", round(rmse_A_ridge, 2))

print("\nMETHOD B")
print("LASSO MAE:", round(mae_B_lasso, 2))
print("RIDGE MAE:", round(mae_B_ridge, 2))
print("LASSO RMSE:", round(rmse_B_lasso, 2))
print("RIDGE RMSE:", round(rmse_B_ridge, 2))

# horizon mae
mae_horizon_lasso = []
mae_horizon_ridge = []

for k in range(1, len(actual_lasso) + 1):
    mae_horizon_lasso.append(mean_absolute_error(actual_lasso[:k], pred_lasso[:k]))
    mae_horizon_ridge.append(mean_absolute_error(actual_ridge[:k], pred_ridge[:k]))

THRESHOLD = 10
valid_lasso = [i + 1 for i, m in enumerate(mae_horizon_lasso) if m <= THRESHOLD]
valid_ridge = [i + 1 for i, m in enumerate(mae_horizon_ridge) if m <= THRESHOLD]

print("\nReliable Horizon (MAE ≤ 10)")
print("LASSO:", max(valid_lasso) if valid_lasso else "None")
print("RIDGE:", max(valid_ridge) if valid_ridge else "None")


# resuls save
pd.DataFrame({
    "actual": y_test.values,
    "predicted_lasso": pred_A_lasso,
    "predicted_ridge": pred_A_ridge
}).to_csv("B_method_A.csv", index=False)

pd.DataFrame({
    "horizon": range(1, len(actual_lasso) + 1),
    "actual": actual_lasso,
    "predicted_lasso": pred_lasso,
    "predicted_ridge": pred_ridge,
    "abs_error_lasso": np.abs(np.array(actual_lasso) - np.array(pred_lasso)),
    "abs_error_ridge": np.abs(np.array(actual_ridge) - np.array(pred_ridge))
}).to_csv("B_method_B.csv", index=False)

pd.DataFrame({
    "horizon": range(1, len(mae_horizon_lasso) + 1),
    "mae_lasso": mae_horizon_lasso,
    "mae_ridge": mae_horizon_ridge
}).to_csv("B_horizon.csv", index=False)

feature_names = poly.get_feature_names_out(X_train.columns)

df_all = pd.DataFrame({
    "Feature": feature_names,
    "Ridge_Coefficient": ridge.coef_,
    "Lasso_Coefficient": lasso.coef_
})

df_all.to_csv("B_coefficients.csv", index=False)


