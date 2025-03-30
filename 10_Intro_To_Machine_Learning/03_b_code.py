#!/usr/bin/env python
"""
This script demonstrates several scikit-learn techniques:
1. Training a Linear Regression model on a simple synthetic dataset.
2. Evaluating it using cross-validation.
3. Hyperparameter tuning with GridSearchCV.
4. Saving and loading the model using joblib.
5. Creating a synthetic dataset for house price prediction.
6. Evaluating model performance using Mean Absolute Error (MAE) and plotting results.
7. Using StandardScaler for feature scaling.
8. Building alternative models (Ridge Regression and RandomForestRegressor).

Note:
- The package is installed as "scikit-learn", but it is imported as "sklearn".
- This is standard practice and scikit-learn is not deprecated.
"""

# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import modules from scikit-learn (officially imported as sklearn)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# For saving and loading the trained model
import joblib

# -------------------------
# SECTION 1: Linear Regression on a Simple Synthetic Dataset
# -------------------------
# We first create a synthetic dataset for a simple linear relationship.
# The relation is: y = 4 + 3x + noise

np.random.seed(42)  # Ensure reproducibility

# Create 100 random points for X in the range [0, 2)
X = 2 * np.random.rand(100, 1)

# Generate y values based on the linear relation with some random noise
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# Split the data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model on the training set.
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Evaluate the model using 5-fold cross-validation on the entire dataset.
cv_scores = cross_val_score(lin_reg, X, y, cv=5)
print("Cross-Validation Score (Linear Regression): {:.4f}".format(cv_scores.mean()))

# Hyperparameter tuning using GridSearchCV:
# We tune the 'fit_intercept' parameter for the Linear Regression model.
param_grid = {'fit_intercept': [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid.fit(X, y)
print("Best parameters from GridSearchCV for Linear Regression:", grid.best_params_)

# Save the trained model using joblib.
joblib.dump(lin_reg, 'linear_model.pkl')
print("Linear Regression model saved as 'linear_model.pkl'.")

# Load the model (for example, in a Flask app you might load a saved model)
loaded_model = joblib.load('linear_model.pkl')
print("Loaded Linear Regression model from 'linear_model.pkl'.")

# -------------------------
# SECTION 2: House Price Prediction Example
# -------------------------
# We generate a synthetic dataset for house price prediction.
# Here, house prices are roughly proportional to the house size with some noise.

np.random.seed(42)  # Re-seed for reproducibility

# Generate random house sizes between 1000 and 3000 square feet.
house_size = np.random.randint(1000, 3000, 50)
# Generate house prices based on size with some noise (price per sq ft = $150 plus random variation)
house_price = house_size * 150 + np.random.randint(-20000, 20000, 50)

# Create a pandas DataFrame with the generated data.
df = pd.DataFrame({'Size': house_size, 'Price': house_price})

# Define features (X) and target (y).
# Note: X needs to be 2D for scikit-learn, so we use double brackets.
X_house = df[['Size']]
y_house = df['Price']

# Split the house data into training (80%) and testing (20%) sets.
X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(X_house, y_house, test_size=0.2, random_state=42)

# Train a Linear Regression model on the house data.
house_model = LinearRegression()
house_model.fit(X_house_train, y_house_train)

# Use the trained model to predict prices for the test set.
predictions = house_model.predict(X_house_test)

# Evaluate the model performance using Mean Absolute Error (MAE).
mae = mean_absolute_error(y_house_test, predictions)
print("Mean Absolute Error for House Price Prediction (Linear Regression): {:.2f}".format(mae))

# Plot the Actual vs. Predicted House Prices.
plt.figure(figsize=(8, 6))
plt.scatter(X_house_test, y_house_test, color='blue', label="Actual Prices")
plt.plot(X_house_test, predictions, color='red', linewidth=2, label="Predicted Prices")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.show()

# -------------------------
# SECTION 3: Data Scaling and Alternative Models
# -------------------------
# In this section, we demonstrate:
# - Feature scaling using StandardScaler
# - Building alternative regression models (Ridge Regression and RandomForestRegressor)

# Standardize the house size feature.
scaler = StandardScaler()
X_house_scaled = scaler.fit_transform(X_house)
print("First 5 rows of scaled house sizes:\n", X_house_scaled[:5])

# --- Alternative Model 1: Ridge Regression ---
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_house_train, y_house_train)
ridge_predictions = ridge_model.predict(X_house_test)
ridge_mae = mean_absolute_error(y_house_test, ridge_predictions)
print("Mean Absolute Error for Ridge Regression: {:.2f}".format(ridge_mae))

# --- Alternative Model 2: Random Forest Regression ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_house_train, y_house_train)
rf_predictions = rf_model.predict(X_house_test)
rf_mae = mean_absolute_error(y_house_test, rf_predictions)
print("Mean Absolute Error for Random Forest Regression: {:.2f}".format(rf_mae))



# Detailed Explanation of the Code Sections:
# Section 1: Linear Regression on a Synthetic Dataset

# Data Creation: We generate 100 random data points with a known linear relationship 
# 𝑦
# =
# 4
# +
# 3
# 𝑥
# y=4+3x and add some random noise.

# Splitting Data: The dataset is split into training and testing subsets using an 80/20 ratio.

# Model Training: A LinearRegression model is trained on the training set.

# Evaluation: We evaluate the model using 5-fold cross-validation.

# Hyperparameter Tuning: We use GridSearchCV to tune the fit_intercept parameter of the model.

# Persistence: The trained model is saved to disk using joblib and then loaded back.

# Section 2: House Price Prediction Example

# Data Generation: We generate synthetic data for house sizes (in square feet) and corresponding prices.

# DataFrame Creation: The data is stored in a pandas DataFrame.

# Model Training & Evaluation: A LinearRegression model is trained on the house data. Predictions are made and evaluated using the Mean Absolute Error (MAE). Finally, the results are plotted.

# Section 3: Data Scaling and Alternative Models

# StandardScaler: We standardize the house size feature for better model performance in some cases.

# Ridge Regression: We build a Ridge regression model and evaluate its performance.

# Random Forest Regression: We build and evaluate a Random Forest regression model.