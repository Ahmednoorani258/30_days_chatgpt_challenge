# Exercise
# Load a dataset (e.g., Titanic from Kaggle: titanic.csv).
# Check for missing values and fill missing Age with the median.
# Compute summary stats for Age and Fare.
# Group by Pclass (passenger class) and find the average Survived rate.

import pandas as pd

# Load a dataset (e.g., Titanic from Kaggle: titanic.csv).
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# Check for missing values and fill missing Age with the median.
print(df.isnull().sum())
df["Age"].fillna(df["Age"].median(), inplace=True)
print(df.isnull().sum())


# Compute summary stats for Age and Fare.
summary = df[["Age", "Fare"]].describe()
print(summary)

# Group by Pclass (passenger class) and find the average Survived rate.
group = df.groupby("Pclass")["Survived"].mean()
print(group)
