# 1. Setup and Import Libraries
# To begin, we need to import the essential Python libraries that will power our machine learning workflow.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# Explanation

# NumPy (np): Handles numerical operations and array manipulations.
# Pandas (pd): Manages data in a tabular format (DataFrames) for easy exploration.
# Matplotlib (plt): Creates visualizations like scatter plots to interpret results.

# Scikit-Learn Modules:

# train_test_split: Splits data into training and testing sets.
# LinearRegression: A simple regression model we’ll use today.
# mean_absolute_error and r2_score: Metrics to evaluate model performance.
# fetch_california_housing: Loads the California Housing dataset.
# Note: The Boston Housing dataset is deprecated due to ethical concerns in newer Scikit-Learn versions. We’ll use the California Housing dataset instead, which is similar and suitable for regression tasks.

# Load the dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

# Preview the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Explanation
# fetch_california_housing(): Downloads the dataset, which includes features and a target variable.
# pd.DataFrame(): Combines the feature data (e.g., median income, house age) with column names.
# df['PRICE']: Adds the target variable (median house value in $100,000s) as a column.
# df.head(): Displays the first 5 rows to get a feel for the data.
# df.describe(): Shows statistics like mean, min, and max for each column.
# df.isnull().sum(): Confirms there are no missing values (important for preprocessing).


    # Insights
    # Features: Include MedInc (median income), HouseAge, AveRooms (average number of rooms), etc.
    # Target (PRICE): Median house value in $100,000s (e.g., 4.526 = $452,600).
    # No Missing Values: The dataset is clean, so we can proceed without imputation.