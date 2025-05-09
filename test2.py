import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * (X.flatten() ** 2) + np.random.normal(0, 0.1, size=X.shape[0])
model = LinearRegression()
model.fit(X, y)

X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training data', s=20)
plt.plot(X_test, y_pred, color='red', label='Predicted function', linewidth=2)

y_true = np.sin(X_test).flatten() + 0.1 * (X_test.flatten() ** 2)
plt.plot(X_test, y_true, color='green', linestyle='--', label='True function')

plt.xlabel('X')
plt.ylabel('f(x)')
plt.title('Prediction of f(x) = sin(x) + 0.1x²')
plt.legend()
plt.grid(True)
plt.show()

x_value = 2
predicted = model.predict(np.array([[x_value]]))
print(f"Predicted value at x={x_value}: {predicted[0]:.4f}")

mae = mean_absolute_error(y, model.predict(X))
mse = mean_squared_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R): {r2:.4f}")