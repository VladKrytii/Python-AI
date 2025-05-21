import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

df = pd.read_csv("internship_candidates_final_numeric.csv")


english_mapping = {
    "Elementary": 1,
    "Pre-Intermediate": 2,
    "Intermediate": 3,
    "Upper-Intermediate": 4,
    "Advanced": 5,
    "Proficient": 6
}

df["EnglishLevelNum"] = df["EnglishLevel"].map(english_mapping)

features = ["Experience", "Grade", "EnglishLevelNum", "Age", "EntryTestScore"]
X = df[features]
y = df["Accepted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
