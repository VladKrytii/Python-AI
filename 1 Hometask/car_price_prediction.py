import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

df = pd.read_csv('cars.csv')

print("Форма набору даних:", df.shape)
print("\nПерші 5 рядків:")
print(df.head())
print("\nСтатистичний опис:")
print(df.describe())
print("\nПеревірка на пропущені значення:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
plt.title('Розподіл цін автомобілів')
plt.xlabel('Ціна (грн)')
plt.ylabel('Кількість')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('price_distribution.png')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Кореляційна матриця')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

X = df[['year', 'engine_volume', 'mileage', 'horsepower']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nОцінка моделі:")
print(f"Середньоквадратична помилка (MSE): {mse:.2f}")
print(f"Корінь з середньоквадратичної помилки (RMSE): {rmse:.2f}")
print(f"Коефіцієнт детермінації (R²): {r2:.4f}")

mean_price = np.mean(y_test)
error_percentage = (rmse / mean_price) * 100
print(f"Середній відсоток помилки: {error_percentage:.2f}%")

coefficients = pd.DataFrame({'Змінна': X.columns, 'Коефіцієнт': model.coef_})
print("\nКоефіцієнти моделі:")
print(coefficients)
print(f"Перетин (intercept): {model.intercept_:.2f}")

plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Справжня vs Прогнозована ціна')
plt.xlabel('Справжня ціна (грн)')
plt.ylabel('Прогнозована ціна (грн)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate(f'R² = {r2:.4f}\nПомилка = {error_percentage:.2f}%', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')

plt.figure(figsize=(12, 8))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Графік залишків')
plt.xlabel('Прогнозована ціна (грн)')
plt.ylabel('Залишки')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('residuals.png')

features = ['year', 'engine_volume', 'mileage', 'horsepower']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.regplot(x=df[feature], y=df['price'], ax=axes[i], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[i].set_title(f'Залежність ціни від {feature}')
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('feature_relationships.png')

print("\nАналіз завершено! Графіки збережено.")