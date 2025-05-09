import pandas as pd
import matplotlib.pyplot as plt

data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"]
}


df = pd.DataFrame(data)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

df["TotalAmount"] = df["Quantity"] * df["Price"]

total_income = df["TotalAmount"].sum()
print(f"Сумарний дохід магазину: {total_income}")

mean_total = df["TotalAmount"].mean()
print(f"Середнє значення TotalAmount: {mean_total}")

orders_per_customer = df["Customer"].value_counts()
print("\nКількість замовлень по кожному клієнту:")
print(orders_per_customer)

print("\nЗамовлення з TotalAmount > 500:")
print(df[df["TotalAmount"] > 500])

sorted_df = df.sort_values(by="OrderDate", ascending=False)
print("\nЗамовлення за спаданням дати:")
print(sorted_df)

mask = (df["OrderDate"] >= "2023-06-05") & (df["OrderDate"] <= "2023-06-10")
print("\nЗамовлення з 5 по 10 червня:")
print(df[mask])

grouped = df.groupby("Category").agg({"Quantity": "sum", "TotalAmount": "sum"})
print("\nГрупування по категоріях:")
print(grouped)

top_customers = df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3)
print("\nТОП-3 клієнтів за TotalAmount:")
print(top_customers)


# Task 2

orders_by_date = df["OrderDate"].value_counts().sort_index()
orders_by_date.plot(kind="line", marker="o", title="Кількість замовлень по датах")
plt.xlabel("Дата")
plt.ylabel("Кількість замовлень")
plt.grid(True)
plt.tight_layout()
plt.show()

df.groupby("Category")["TotalAmount"].sum().plot(kind="bar", title="Доходи по категоріях", color='orange')
plt.xlabel("Категорія")
plt.ylabel("TotalAmount")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
