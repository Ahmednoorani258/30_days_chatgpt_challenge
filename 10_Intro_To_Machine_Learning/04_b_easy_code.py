#!/usr/bin/env python
"""
Simple Linear Regression Example for House Price Prediction
-------------------------------------------------------------
This script shows you how to:

1. Create a simple dataset (house sizes and prices).
2. Split the data into training and testing sets.
3. Train a Linear Regression model.
4. Make predictions.
5. Evaluate the model using Mean Absolute Error (MAE).
6. Plot the results (Actual vs. Predicted prices).

We use plain language and clear comments to help you understand each step.
"""

# Step 1: Import required libraries
import numpy as np          # For numerical calculations
import pandas as pd         # For data handling
import matplotlib.pyplot as plt  # For plotting graphs

# Import the Linear Regression model and a function to split data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 2: Create a simple synthetic dataset
# We will create data for house sizes and prices.
# Let's assume:
#   House Price = (House Size in sq ft) * 150 + some random noise

# Set a seed so that the random numbers are the same every time
np.random.seed(42)

# Create 50 random house sizes between 1000 and 3000 square feet
house_sizes = np.random.randint(1000, 3000, 50)

# Create house prices with a formula and random noise
house_prices = house_sizes * 150 + np.random.randint(-20000, 20000, 50)

# Create a DataFrame to hold the data (for easier viewing and handling)
df = pd.DataFrame({'Size': house_sizes, 'Price': house_prices})
print("First 5 rows of our data:")
print(df.head())

# Step 3: Prepare the data for scikit-learn
# Scikit-learn expects the features (X) to be a 2D array.
X = df[['Size']]   # Features (house sizes)
y = df['Price']    # Target (house prices)

# Step 4: Split the data into training and testing sets
# We'll use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nNumber of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])

# Step 5: Create and train the Linear Regression model
model = LinearRegression()  # Create the model
model.fit(X_train, y_train)  # Train the model on the training data
print("\nLinear Regression model has been trained.")

# Step 6: Make predictions on the test set
predictions = model.predict(X_test)

# Step 7: Evaluate the model performance
# We use Mean Absolute Error (MAE) to see how far off our predictions are.
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error (MAE) of the model: {:.2f}".format(mae))

# Step 8: Plot Actual Prices vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label="Actual Prices")  # Plot actual prices
plt.plot(X_test, predictions, color='red', linewidth=2, label="Predicted Prices")  # Plot predictions as a line
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()

"""
Summary:
--------
1. We generated random data for house sizes and calculated prices.
2. The data was split into training and testing sets.
3. A Linear Regression model was trained on the training data.
4. We used the model to predict house prices on the test set.
5. The model's error (MAE) was calculated to check performance.
6. Finally, we plotted the actual vs. predicted prices for visualization.
"""
