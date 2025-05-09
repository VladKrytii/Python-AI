import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1
plt.figure(figsize=(10, 6))
x = np.linspace(-10, 10, 500)
y = x**4 * np.sin(x)
plt.plot(x, y, label='$f(x) = x^4 \sin(x)$', color='blue')
plt.title('Графік функції $f(x) = x^4 \sin(x)$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# 2
data = np.random.normal(loc=3, scale=2, size=1000)
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True, color='green', bins=30)
plt.title('Гістограма нормального розподілу (μ=3, σ=2)')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# 3
hobbies = ['Читання', 'Спорт', 'Музика', 'Подорожі', 'Програмування']
frequencies = [20, 15, 25, 30, 10]
plt.figure(figsize=(8, 8))
plt.pie(frequencies, labels=hobbies, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Розподіл часу на хобі')
plt.show()

# 4
np.random.seed(42)
forest_types = ['Сосна', 'Дуб', 'Береза', 'Ялина']
data_forests = [np.random.normal(loc=i*10, scale=2, size=100) for i in range(1, 5)]
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_forests, palette='Set2')
plt.xticks(range(4), forest_types)
plt.title('Розподіл маси дерев для 4 типів лісів')
plt.xlabel('Тип лісу')
plt.ylabel('Маса (кг)')
plt.grid(True)
plt.show()

# 5
x_uniform = np.random.uniform(0, 1, 100)
y_uniform = np.random.uniform(0, 1, 100)
plt.figure(figsize=(10, 6))
plt.scatter(x_uniform, y_uniform, color='purple', alpha=0.6, s=80)
plt.title('Точкова діаграма для рівномірного розподілу')
plt.xlabel('Вісь X')
plt.ylabel('Вісь Y')
plt.grid(True)
plt.show()

# 6
x = np.linspace(0, 2*np.pi, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='$\sin(x)$', color='red', linestyle='-')
plt.plot(x, np.cos(x), label='$\cos(x)$', color='blue', linestyle='--')
plt.plot(x, np.sin(x) + np.cos(x), label='$\sin(x) + \cos(x)$', color='green', linestyle='-.')
plt.title('Графіки функцій')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()