import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# data
train = pd.read_csv("train.csv", index_col=0, parse_dates=True)
val = pd.read_csv("validation.csv", index_col=0, parse_dates=True)

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_val = val.drop(columns=["close", "Year", "Month"])
y_val = val["close"]

# model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluation
def evaluate(y, y_pred, name):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

evaluate(y_train, model.predict(X_train), "TRAINING")
evaluate(y_val, model.predict(X_val), "VALIDATION")

print("\nΣυντελεστές μοντέλου:")
print(pd.Series(model.coef_, index=X_train.columns))
