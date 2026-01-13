import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


# Φορτωση δεδομενων 
df = pd.read_csv("AMGN_monthly.csv", index_col=0, parse_dates=True)

df.sort_index(inplace=True)
df["Year"] = df.index.year
df["Month"] = df.index.month

# Gaussian smoothing
SIGMA = 1
df["close"] = gaussian_filter1d(df["close"], sigma=SIGMA)

# features με lags 
N = 3
for i in range(1, N+1):
    df[f"close_t-{i}"] = df["close"].shift(i)
    df[f"volume_t-{i}"] = df["volume"].shift(i)

df.dropna(inplace=True)

# Διαχωρισμος δεδομενων training/validation

train = df[df["Year"] < 2024]
val = df[df["Year"] >= 2024]

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]
X_val = val.drop(columns=["close", "Year", "Month"])
y_val = val["close"]


train.to_csv("train.csv")
val.to_csv("validation.csv")

#μετρικες
def evaluate(y, y_pred, label):
    print("\n", label)
    print("MAE:", mean_absolute_error(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

