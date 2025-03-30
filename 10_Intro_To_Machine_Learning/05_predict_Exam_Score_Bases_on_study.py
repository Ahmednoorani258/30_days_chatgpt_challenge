# Task: Predict Student Exam Scores Based on Study Hours
# Objective
# Build a simple Linear Regression model to predict a student’s final exam score (a continuous value) based on the number of hours they studied. This task uses supervised learning with a small synthetic dataset, making it beginner-friendly while hitting all the key concepts from Day 10.


#  Step 1 Setup Environmet
# _____________________________________________________________

# What You’ll Need
# Python installed (3.7 or higher).
# Libraries: numpy, pandas, matplotlib, scikit-learn.
# A code editor (e.g., VS Code, Jupyter Notebook).

# run this command
# pip install numpy pandas matplotlib scikit-learn


# Step 2 Data Collection and Preparation
# _____________________________________________________________


import numpy as np
import pandas as pd


np.random.seed(42)  # For reproducibility

study_hours = np.random.randint(0, 10, 20 ) # 20 students, 1-10 hours
exam_scores = study_hours * 9 + np.random.uniform(-10, 10, 20) # Score = 9 * hours + noise

# print(study_hours)
# print(exam_scores)

# Create a DataFrame
data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Exam_Score': exam_scores
})

# Check for missing values (none in this case, but good practice)
print("Missing Values:\n", data.isnull().sum())

# Basic data exploration
print("\nData Preview:\n", data.head(20))
print("\nSummary Statistics:\n", data.describe())


"""
Guidelines
Data Collection: We’re simulating data since this is a basic task. In real-world scenarios, you’d collect data from a CSV, database, or API.
Data Cleaning: Check for missing values with isnull().sum(). Here, our synthetic data is clean, but always verify.
Feature Engineering: We’re using "Study_Hours" as the feature and "Exam_Score" as the target. No extra transformation is needed for simplicity.
Why
Demonstrates data preparation: generating, structuring, and checking data.
Shows how features (inputs) and targets (outputs) are defined in supervised learning.
"""




# _____________________________________________________________
# Step 3 Model Selection
# _____________________________________________________________


# Task
# Choose Linear Regression as the algorithm since we’re predicting a continuous value (exam score).

from sklearn.linear_model import LinearRegression

# Define feature (X) and target (y)
X = data[['Study_Hours']]  # 2D array for scikit-learn
y = data['Exam_Score']

# Initialize the model
model = LinearRegression()

"""
Guidelines
Why Linear Regression: It’s simple, interpretable, and fits our problem (predicting a number based on another number).
X as 2D: Scikit-learn expects features in a 2D format ([[value1], [value2], ...]), so we use double brackets.
Why
Introduces model selection: picking an algorithm suited to the task (regression here).
Keeps it basic while showing how to instantiate a model.
"""


# _____________________________________________________________
# Step 4 Model Training
# _____________________________________________________________

# Task
# Split the data into training and testing sets, then train the Linear Regression model.

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model.fit(X_train, y_train)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Print the learned parameters
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

"""
Guidelines
Splitting Data: Use 80% for training (test_size=0.2 means 20% for testing). random_state=42 ensures consistent splits.
Training: fit() adjusts the model’s slope and intercept to minimize prediction error.
Parameters: The slope (coefficient) shows how much the score increases per hour studied, and the intercept is the base score.
Why
Teaches data splitting to evaluate generalization.
Shows how a model learns from data (supervised learning in action).
"""

# _____________________________________________________________
# Step 5 Model Evaluation
# _____________________________________________________________

# Task
# Make predictions on the test set and evaluate the model using Mean Absolute Error (MAE).

from sklearn.metrics import mean_absolute_error

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

"""
Guidelines
Predictions: Use predict() to estimate scores for test hours.
MAE: Measures average prediction error in the same units as the target (exam points). Lower is better.
Interpret: If MAE is 5, predictions are off by 5 points on average.
Why
Demonstrates evaluation: how to measure model performance.
Introduces a key metric (MAE) from the ML pipeline.
"""

# ___________________________________________________________
# Step 6 Visualization
# _____________________________________________________________

# Task
# Plot actual vs. predicted scores to see how well the model fits the data.

import matplotlib.pyplot as plt

# Scatter plot of actual vs predicted
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')
plt.plot(X_test, y_pred, color='red', label='Predicted Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Linear Regression: Exam Score Prediction')
plt.legend()

# Display in Streamlit (or locally)
# st.pyplot(plt)  # Use this in Streamlit; otherwise, use plt.show()
plt.show()

"""
Guidelines
Scatter Plot: Blue dots are actual test data points.
Line Plot: Red line is the model’s predictions, showing the learned relationship.
Streamlit: If using Streamlit, replace plt.show() with st.pyplot(plt).
Why
Visualizes model fit: see how close predictions are to reality.
Reinforces supervised learning by showing input-output mapping.
"""