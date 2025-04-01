import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate with Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# Evaluate with R² Score
r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.2f}")

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Prices ($100,000s)")
plt.ylabel("Predicted Prices ($100,000s)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()