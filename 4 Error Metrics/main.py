import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1
X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * X.flatten() ** 2

# 2
model = LinearRegression()
model.fit(X, y)

# 3
y_pred = model.predict(X)

# 4
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 5
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Реальна функція', color='blue')
plt.plot(X, y_pred, label='Прогноз моделі', color='red', linestyle='--')
plt.title("Передбачення функції f(x) = sin(x) + 0.1x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# 6
x_val = 7
y_true = np.sin(x_val) + 0.1 * x_val**2
y_model = model.predict(np.array([[x_val]]))[0]

print(f"\nРеальне значення f({x_val}) = {y_true:.4f}")
print(f"Передбачене значення моделі f({x_val}) = {y_model:.4f}")
