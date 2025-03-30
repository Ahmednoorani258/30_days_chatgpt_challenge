#!/usr/bin/env python
"""
Extended Script Demonstrating scikit‑learn Techniques with Detailed Explanations
-----------------------------------------------------------------------------------
This script covers:

SECTION 1: Linear Regression on a Synthetic Dataset
    - Creating a synthetic dataset with a known linear relationship.
    - Splitting the dataset into training and testing sets.
    - Training a Linear Regression model.
    - Evaluating the model using cross-validation.
    - Hyperparameter tuning with GridSearchCV.
    - Saving and loading the model with joblib.
    
SECTION 2: House Price Prediction Example
    - Generating a synthetic dataset for predicting house prices.
    - Visualizing the actual vs. predicted prices.
    - Computing evaluation metrics (Mean Absolute Error).

SECTION 3: Data Scaling and Alternative Models
    - Standardizing features with StandardScaler.
    - Building and evaluating a Ridge Regression model.
    - Building and evaluating a Random Forest Regressor.

NOTE:
    - Although the package is installed as "scikit‑learn", we import it as "sklearn".
    - This is the official import name and is not deprecated.
    
Enjoy the detailed examples and explanations in the comments!
"""

# ============================
# IMPORT NECESSARY PACKAGES
# ============================
import numpy as np          # For numerical operations
import pandas as pd         # For DataFrame handling
import matplotlib.pyplot as plt  # For plotting

# Import modules from scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# For saving and loading models
import joblib

# ============================
# SECTION 1: Linear Regression on a Synthetic Dataset
# ============================

# -- Data Generation --
# We create a simple linear dataset where:
#   y = 4 + 3*x + (random noise)
np.random.seed(42)  # Seed for reproducibility
X = 2 * np.random.rand(100, 1)  # Generate 100 values between 0 and 2
# Generate target variable y with added noise from a normal distribution
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# -- Splitting the Data --
# We use an 80/20 split for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("SECTION 1: Linear Regression on Synthetic Data")
print("Training size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -- Model Training --
# Create a Linear Regression model and train (fit) it on the training set.
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print("Linear Regression model trained.")

# -- Cross-Validation --
# Evaluate the model using 5-fold cross-validation over the entire dataset.
cv_scores = cross_val_score(lin_reg, X, y, cv=5)
print("Cross-Validation Score (Linear Regression): {:.4f}".format(cv_scores.mean()))

# -- Hyperparameter Tuning --
# Here we use GridSearchCV to determine whether including an intercept is beneficial.
param_grid = {'fit_intercept': [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid.fit(X, y)
print("Best parameters from GridSearchCV for Linear Regression:", grid.best_params_)

# -- Model Persistence --
# Save the trained model using joblib for later use (e.g., in a Flask application).
joblib.dump(lin_reg, 'linear_model.pkl')
print("Linear Regression model saved as 'linear_model.pkl'.")

# Load the model back from file to ensure it works.
loaded_model = joblib.load('linear_model.pkl')
print("Loaded Linear Regression model from 'linear_model.pkl'.\n")

# ============================
# SECTION 2: House Price Prediction Example
# ============================

# -- Data Generation --
# We generate synthetic data for house prices:
#   House Price ≈ House Size (sq ft) * 150 + noise
np.random.seed(42)  # Re-seed for reproducibility

# Generate 50 random house sizes between 1000 and 3000 square feet.
house_size = np.random.randint(1000, 3000, 50)
# Create house prices with some noise: base price = size * 150 with random variation.
house_price = house_size * 150 + np.random.randint(-20000, 20000, 50)

# Create a DataFrame for better visualization and handling.
df = pd.DataFrame({'Size': house_size, 'Price': house_price})
print("House Price Data (first 5 rows):\n", df.head(), "\n")

# -- Define Features and Target --
# X needs to be a 2D array, so we use double brackets.
X_house = df[['Size']]
y_house = df['Price']

# -- Splitting the Data --
# Split into training and testing sets (80/20 split).
X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42)
print("House Price Prediction: Training size:", X_house_train.shape[0],
      "Test size:", X_house_test.shape[0])

# -- Model Training --
# Train a Linear Regression model on the house data.
house_model = LinearRegression()
house_model.fit(X_house_train, y_house_train)
print("House price Linear Regression model trained.")

# -- Prediction and Evaluation --
# Predict the house prices on the test set.
predictions = house_model.predict(X_house_test)
# Calculate Mean Absolute Error (MAE) to evaluate model performance.
mae = mean_absolute_error(y_house_test, predictions)
print("Mean Absolute Error (Linear Regression, House Prices): {:.2f}".format(mae))

# -- Plotting --
# Visualize the relationship between actual and predicted house prices.
plt.figure(figsize=(8, 6))
plt.scatter(X_house_test, y_house_test, color='blue', label="Actual Prices")
# Note: Since X_house_test is 2D, plt.plot will plot the line by connecting points in order.
plt.plot(X_house_test, predictions, color='red', linewidth=2, label="Predicted Prices")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.show()

# ============================
# SECTION 3: Data Scaling and Alternative Models
# ============================

# -- Data Scaling --
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X_house_scaled = scaler.fit_transform(X_house)
print("First 5 rows of scaled house sizes:\n", X_house_scaled[:5], "\n")

# -- Alternative Model 1: Ridge Regression --
# Ridge Regression is similar to Linear Regression but includes L2 regularization.
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_house_train, y_house_train)
ridge_predictions = ridge_model.predict(X_house_test)
ridge_mae = mean_absolute_error(y_house_test, ridge_predictions)
print("Mean Absolute Error for Ridge Regression: {:.2f}".format(ridge_mae))

# -- Alternative Model 2: Random Forest Regression --
# RandomForestRegressor is an ensemble method that builds multiple decision trees.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_house_train, y_house_train)
rf_predictions = rf_model.predict(X_house_test)
rf_mae = mean_absolute_error(y_house_test, rf_predictions)
print("Mean Absolute Error for Random Forest Regression: {:.2f}".format(rf_mae))

# ============================
# END OF SCRIPT
# ============================

"""
Detailed Explanation Recap:
----------------------------------
SECTION 1:
- We generated a synthetic linear dataset and trained a Linear Regression model.
- Cross-validation provided an estimate of model performance.
- GridSearchCV helped tune the model's hyperparameters (fit_intercept).
- The model was saved and later reloaded using joblib.

SECTION 2:
- Synthetic house price data was created, where prices are based on house sizes.
- A Linear Regression model was trained on the house data and evaluated using MAE.
- The results were visualized by plotting actual vs. predicted prices.

SECTION 3:
- Feature scaling was performed using StandardScaler to standardize house sizes.
- Two alternative models (Ridge Regression and RandomForestRegressor) were built.
- Their performance was compared using Mean Absolute Error.
  
This script demonstrates a full workflow from data creation, model training, evaluation, and saving/loading,
to comparing multiple models. Enjoy experimenting with and modifying this code for your own projects!
"""
