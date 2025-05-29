import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
df = pd.DataFrame({"Date": dates})
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.weekday
df["IsWeekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

df["Temperature"] = 10 + 15 * np.sin((df["DayOfYear"] - 80) * np.pi / 183) + np.random.normal(0, 3, len(df))

df["Consumption"] = (
    50 +
    10 * np.abs(df["Temperature"] - 20) +
    5 * df["IsWeekend"] +
    np.random.normal(0, 5, len(df))
)

X = df[["DayOfYear", "Month", "Weekday", "IsWeekend", "Temperature"]].values
y = df["Consumption"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = Sequential()
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.1)

y_pred = model.predict(X_test).flatten()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Справжнє споживання")
plt.ylabel("Передбачене споживання")
plt.title("Передбачення споживання електроенергії нейромережею")
plt.grid(True)
plt.tight_layout()
plt.show()

sample_dates = pd.to_datetime(["2023-01-01", "2023-07-15", "2023-12-01"])
sample_df = pd.DataFrame({
    "DayOfYear": sample_dates.dayofyear,
    "Month": sample_dates.month,
    "Weekday": sample_dates.weekday,
    "IsWeekend": [1 if d.weekday() >= 5 else 0 for d in sample_dates],
})

sample_df["Temperature"] = [0, 28, -2]

X_sample = scaler.transform(sample_df)
pred = model.predict(X_sample).flatten()

print("\n--- Прогноз споживання електроенергії ---")
for d, p in zip(sample_dates, pred):
    print(f"{d.date()} — прогнозоване споживання: {p:.2f} кВт⋅год")
