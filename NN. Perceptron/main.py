import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
hours = np.linspace(0, 24, 500)  # 0:00 до 24:00

duration = (
    20 + 10 * np.sin((hours - 8) * np.pi / 12) +
    15 * np.exp(-((hours - 18) ** 2) / 8) +
    np.random.normal(0, 2, len(hours))
)

df = pd.DataFrame({"Hour": hours, "Duration": duration})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[["Hour"]].values
y = df["Duration"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2)


hour_inputs = np.array([[10.5], [0.0], [2.666]])
hour_inputs_scaled = scaler.transform(hour_inputs)
nn_predictions = model.predict(hour_inputs_scaled).flatten()

print("Прогноз нейромережі:")
for h, d in zip(hour_inputs.flatten(), nn_predictions):
    print(f"{h:.2f} годин — {d:.2f} хв.")


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train_p, y_train_p)

poly_predictions = reg.predict(poly.transform(hour_inputs))

print("\nПрогноз поліноміальної регресії:")
for h, d in zip(hour_inputs.flatten(), poly_predictions):
    print(f"{h:.2f} годин — {d:.2f} хв.")

hours_grid = np.linspace(0, 24, 500).reshape(-1, 1)
hours_scaled = scaler.transform(hours_grid)

nn_pred = model.predict(hours_scaled).flatten()
poly_pred = reg.predict(poly.transform(hours_grid))

plt.figure(figsize=(10, 6))
plt.plot(hours, duration, 'o', alpha=0.3, label='Реальні дані')
plt.plot(hours_grid, nn_pred, label='Нейронна мережа', linewidth=2)
plt.plot(hours_grid, poly_pred, label='Поліноміальна регресія', linewidth=2)
plt.xlabel('Час доби (год)')
plt.ylabel('Тривалість поїздки (хв)')
plt.title('Порівняння моделей')
plt.legend()
plt.grid(True)
plt.show()
