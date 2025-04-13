# Histogram with Matplotlib
# A histogram is perfect for visualizing the distribution of a single variable, such as house prices.

# Step 1: Importing Matplotlib
# Matplotlib is Python’s core plotting library. Start by importing its plotting module:

import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Plotting a Histogram
# Assuming you have a DataFrame df loaded with the house_prices.csv data, use plt.hist() to create a histogram of the Price column:

df = pd.read_csv("Housing.csv")
# Histogram of house prices
plt.hist(df["price"], bins=50, color="green")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()


"""
What’s Happening?
plt.hist(df['Price'], bins=20, color='blue'):
df['Price']: The data to plot (house prices).
bins=20: Divides the price range into 20 intervals. More bins provide finer detail, while fewer bins give a broader &oobgshift; a smoother curve.
color='blue': Sets the bars to blue for visual appeal.
plt.xlabel('Price'): Labels the x-axis as "Price."
plt.ylabel('Frequency'): Labels the y-axis as "Frequency" (how many houses fall into each bin).
plt.title('Distribution of House Prices'): Adds a title.
plt.show(): Displays the plot.
Insight
The histogram shows the shape of the data:

Is it normal (bell-shaped), skewed (e.g., most homes are cheap), or bimodal (two peaks)?
Are there gaps or clusters in the price range?
Key Point
Histograms are ideal for understanding the spread, central tendency, and overall distribution of a single variable.
"""
