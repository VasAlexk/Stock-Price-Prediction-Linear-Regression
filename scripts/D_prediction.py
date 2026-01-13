import pandas as pd
from sklearn.linear_model import LinearRegression

# Φόρτωση δεδομένων 
df = pd.read_csv("AMGN_monthly.csv", index_col=0, parse_dates=True)
df.sort_index(inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month

# Lags
N = 3
for i in range(1, N + 1):
    df[f"close_t-{i}"] = df["close"].shift(i)
    df[f"volume_t-{i}"] = df["volume"].shift(i)

df.dropna(inplace=True)

# Training: μέχρι 2023 
train = df[df["Year"] < 2024]
X_train = train.drop(columns=["close", "Year", "Month"])
y_train = train["close"]

model = LinearRegression()
model.fit(X_train, y_train)

# Βρίσκουμε τη γραμμή για Δεκέμβριο 2024
start_row = df[(df["Year"] == 2024) & (df["Month"] == 12)].copy()

if start_row.empty:
    raise ValueError("Δεν βρέθηκαν δεδομένα για Δεκέμβριο 2024 στο AMGN_monthly.csv")

current = start_row.copy()
current_date = current.index[0]

jan_2025 = None
dec_2025 = None

# Προβλέπουμε διαδοχικά μήνα-μήνα
for step in range(1, 13): 
    # Ημερομηνία επόμενου μήνα
    current_date = (current_date + pd.offsets.DateOffset(months=1))

    # Πρόβλεψη τιμής close 
    y_pred = model.predict(current[X_train.columns])[0]

    # Αποθήκευση για τουτς μηνες που θελουμε
    if current_date.year == 2025 and current_date.month == 1:
        jan_2025 = y_pred
    if current_date.year == 2025 and current_date.month == 12:
        dec_2025 = y_pred

    #  ανανέωση lags της close 
    new = current.copy()
    new["close"] = y_pred

    for j in reversed(range(2, N + 1)):
        new[f"close_t-{j}"] = new[f"close_t-{j-1}"]
    new["close_t-1"] = y_pred

    current = new

# results
print("ΠΡΟΒΛΕΨΕΙΣ")
if jan_2025 is not None:
    print("Ιανουάριος 2025:", round(jan_2025, 2))
else:
    print("Δεν υπολογίστηκε πρόβλεψη ")

if dec_2025 is not None:
    print("Δεκέμβριος 2025:", round(dec_2025, 2))
else:
    print("Δεν υπολογίστηκε πρόβλεψη ")
