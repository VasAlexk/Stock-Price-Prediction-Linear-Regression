import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# data
train = pd.read_csv("train.csv", index_col=0, parse_dates=True)
val = pd.read_csv("validation.csv", index_col=0, parse_dates=True)

X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

X_val = val.drop(columns=["close", "Year", "Month"])
y_val = val["close"]

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# PCA me 95% variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# model
model = LinearRegression()
model.fit(X_train_pca, y_train)

# prediction
train_pred = model.predict(X_train_pca)
val_pred = model.predict(X_val_pca)

# metrikes
train_mae = mean_absolute_error(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

val_mae = mean_absolute_error(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

# apotelesmata
print("\nPCA RESULTS")

print("\nTRAIN METRICS")
print("MAE :", round(train_mae, 4))
print("RMSE:", round(train_rmse, 4))

print("\nVALIDATION METRICS")
print("MAE :", round(val_mae, 4))
print("RMSE:", round(val_rmse, 4))

print("\nPCA COMPONENTS INFO")
print("Number of components:", pca.n_components_)

print("\nExplained variance ratio per component:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    print(f"PC{i}: {var:.4f}")

print("\nCumulative explained variance:",
      round(np.sum(pca.explained_variance_ratio_), 4))

print("\nPCA COMPONENT MATRIX (Eigenvectors):")
component_names = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
components_df = pd.DataFrame(
    pca.components_,
    index=component_names,
    columns=X_train.columns
)
print(components_df.round(3))

# save se csv
components_df.to_csv("C_PCA_components.csv")


