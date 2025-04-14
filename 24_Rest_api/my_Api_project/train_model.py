# train_model.py
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# Fake data: house sizes (square feet) and prices (dollars)
X = np.array([[1000], [1500], [2000]])  # Sizes
y = np.array([100000, 150000, 200000])  # Prices

# Train a simple model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, "model/model.joblib")
print("Model saved!")