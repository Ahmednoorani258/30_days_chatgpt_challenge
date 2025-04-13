import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Set up the app
st.title("🧠 Student Exam Score Prediction with Linear Regression")
st.write(
    """
This app demonstrates the machine learning pipeline using linear regression to predict exam scores based on study hours.
Follow along to see each step in action!
"""
)

# Step 2: Data Generation and Preparation
st.header("1. Data Generation and Preparation")
st.write(
    """
We'll generate synthetic data for study hours and exam scores. You can adjust the number of students and the range of study hours.
"""
)

# User inputs for data generation
num_students = st.slider("Number of Students", min_value=10, max_value=100, value=20)

min_hours = st.number_input(
    "Minimum Study Hours", min_value=1.0, max_value=5.0, value=1.0
)
max_hours = st.number_input(
    "Maximum Study Hours", min_value=5.0, max_value=10.0, value=10.0
)

# Generate synthetic data
np.random.seed(42)  # For reproducibility
study_hours = np.random.uniform(min_hours, max_hours, num_students)
exam_scores = study_hours * 9 + np.random.uniform(
    -10, 10, num_students
)  # Linear relationship with noise

# Create DataFrame
data = pd.DataFrame({"Study_Hours": study_hours, "Exam_Score": exam_scores})


# Display data preview
st.write("### Data Preview")
st.dataframe(data)
# Step 3: Model Selection
st.header("2. Model Selection")
st.write(
    """
We choose **Linear Regression** because it's simple and effective for predicting a continuous value (exam score) based on another (study hours).
"""
)
st.code(
    "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()",
    language="python",
)

# Step 4: Model Training
st.header("3. Model Training")
st.write(
    """
We split the data into training and testing sets, then train the model on the training data.
"""
)

# Define features and target
X = data[["Study_Hours"]]
y = data["Exam_Score"]

# Allow user to adjust test size
test_size = st.slider("Test Size (percentage)", min_value=0.1, max_value=0.5, value=0.2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display learned parameters
st.write("### Learned Parameters")
st.write(f"Slope (Coefficient): {model.coef_[0]:.2f}")
st.write(f"Intercept: {model.intercept_:.2f}")
st.write(
    "**Interpretation**: The slope indicates how much the exam score increases per additional study hour."
)

# Step 5: Model Evaluation
st.header("4. Model Evaluation")
st.write(
    """
We evaluate the model using **Mean Absolute Error (MAE)**, which measures the average prediction error.
"""
)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
# explaination of mae dropdown menu arrow with streamlit

expand = st.expander("MAE Explanation")
expand.write(
    """
            Let me explain the line of code mae = mean_absolute_error(y_test, y_pred) in a clear and straightforward way.

What Does This Code Do?
This line calculates the Mean Absolute Error (MAE), a metric used to evaluate how accurate a regression model's predictions are. It measures the average difference between the actual values and the predicted values, ignoring whether the predictions are too high or too low.

Breaking It Down
1. The Function: mean_absolute_error
This is a function from the sklearn.metrics module in Python's scikit-learn library, a popular tool for machine learning.
It’s designed to compare two sets of numbers: the true values and the predictions.
2. The Inputs
y_test: This is a list or array of the actual values from your test dataset (the real outcomes you’re trying to predict).
y_pred: This is a list or array of the predicted values that your model came up with for the same test dataset.
3. How It Works
The mean_absolute_error function:

Takes each pair of values (one from y_test, one from y_pred).
Calculates the absolute difference between them (so negative errors become positive).
Adds up all these differences and divides by the total number of pairs to find the average.
Mathematically, it’s expressed as:

MAE
=
1
𝑛
∑
𝑖
=
1
𝑛
∣
𝑦
𝑖
−
𝑦
^
𝑖
∣
MAE= 
n
1
​
  
i=1
∑
n
​
 ∣y 
i
​
 − 
y
^
​
  
i
​
 ∣
𝑛
n: Number of data points.
𝑦
𝑖
y 
i
​
 : An actual value from y_test.
𝑦
^
𝑖
y
^
​
  
i
​
 : A predicted value from y_pred.
∣
𝑦
𝑖
−
𝑦
^
𝑖
∣
∣y 
i
​
 − 
y
^
​
  
i
​
 ∣: The absolute difference between the two.
4. The Output
The result is stored in the variable mae.
This number tells you, on average, how far off your predictions are from the actual values, in the same units as your data.
Example
Imagine you’re predicting exam scores:

Actual scores (y_test): [50, 60, 70]
Predicted scores (y_pred): [48, 62, 68]
Here’s the step-by-step calculation:

Difference 1: 
∣
50
−
48
∣
=
2
∣50−48∣=2
Difference 2: 
∣
60
−
62
∣
=
2
∣60−62∣=2
Difference 3: 
∣
70
−
68
∣
=
2
∣70−68∣=2
Sum of differences: 
2
+
2
+
2
=
6
2+2+2=6
Number of data points: 
3
3
MAE: 
6
/
3
=
2
6/3=2
So, mae = 2, meaning the predictions are, on average, 2 points away from the actual scores.

Why Is MAE Useful?
Easy to Understand: It’s in the same units as your data (e.g., if you’re predicting temperatures in Celsius, MAE is in Celsius).
Fair Evaluation: It treats all errors equally by using absolute values, making it less affected by huge mistakes compared to other metrics like Mean Squared Error.
Model Performance: It helps you see how close your model’s predictions are to reality.
In Summary
The line mae = mean_absolute_error(y_test, y_pred) is a simple way to measure the average error of a regression model by calculating the mean of the absolute differences between actual values (y_test) and predicted values (y_pred). It’s a handy tool for checking how well your model is doing!
            """
)
# expand.title("MAE Explanation")

st.write(f"### Mean Absolute Error (MAE): {mae:.2f}")
st.write(
    "**Interpretation**: On average, the model's predictions are off by this many points."
)

# Step 6: Visualize Results
st.header("5. Visualize Results")
st.write(
    """
The scatter plot shows actual exam scores (blue dots) vs. predicted scores (red line).
"""
)

# Create plot
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color="blue", label="Actual Scores")
ax.plot(X_test, y_pred, color="red", label="Predicted Scores")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")
ax.set_title("Linear Regression: Exam Score Prediction")
ax.legend()

# Display plot
st.pyplot(fig)

# Step 7: Conclusion and Next Steps
st.header("6. Conclusion")
st.write(
    """
- **What We Did**: Generated data, trained a linear regression model, evaluated it, and visualized the results.
- **Key Takeaway**: Linear regression can effectively model linear relationships, but real-world data may require more complex models.
"""
)

# Optional: Allow users to download the data
st.write("### Download Data")
st.download_button(
    label="Download CSV",
    data=data.to_csv(index=False),
    file_name="student_data.csv",
    mime="text/csv",
)
