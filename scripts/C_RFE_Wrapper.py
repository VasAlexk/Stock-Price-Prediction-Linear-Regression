import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# data
train = pd.read_csv("train.csv", index_col=0, parse_dates=True)
val = pd.read_csv("validation.csv", index_col=0, parse_dates=True)

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_val = val.drop(columns=["close", "Year", "Month"])
y_val = val["close"]

# feture selection  me rfe
base_model = LinearRegression()
selector = RFE(base_model, n_features_to_select=4)
selector.fit(X_train, y_train)

# Selected features
selected = X_train.columns[selector.support_]

print("\nRFE SELECTED FEATURES:")
for i, f in enumerate(selected, 1):
    print(f"{i}. {f}")

print("\nFeature ranking (1 = selected):")
ranking_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Rank": selector.ranking_
}).sort_values(by="Rank")

print(ranking_df.to_string(index=False))


# training modelou me selected features
model = LinearRegression()
model.fit(X_train[selected], y_train)

# prediction
train_pred = model.predict(X_train[selected])
val_pred = model.predict(X_val[selected])

# metrikes
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    return mae, rmse

print("\nRFE MODEL RESULTS")
train_mae, train_rmse = evaluate("TRAIN SET", y_train, train_pred)
val_mae, val_rmse = evaluate("VALIDATION SET", y_val, val_pred)


# syntelestes montelou
coef_df = pd.DataFrame({
    "Feature": selected,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nRFE MODEL COEFFICIENTS (sorted by importance):")
print(coef_df.to_string(index=False))
