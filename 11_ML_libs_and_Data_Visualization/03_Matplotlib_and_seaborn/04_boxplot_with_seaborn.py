# C. Boxplot with Seaborn
# A boxplot summarizes a variable’s distribution across categories, making it great for comparisons.

# Step 1: Plotting a Boxplot
# Use sns.boxplot() to compare Price across different Bedrooms categories:

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('Housing.csv')
# Boxplot: Price by Bedrooms
sns.boxplot(x='bedrooms', y='price', data=df, palette='Set3')
plt.title('Price Distribution by Bedrooms')
plt.show()

# # Scatter plot with a regression line
# sns.regplot(x='area', y='price', data=df)
# plt.title('House Size vs Price with Trend Line')
# plt.show()

# What’s Happening?
# sns.boxplot(x='Bedrooms', y='Price', data=df'):
# x='Bedrooms': Groups the data by number of bedrooms.
# y='Price': Plots price on the y-axis.
# data=df: Uses the DataFrame df.
# Each box represents the price distribution for a specific bedroom count and shows:
# Median: The line inside the box (middle value).
# Quartiles: The box edges (25th and 75th percentiles).
# Whiskers: Extend to 1.5 times the interquartile range (IQR), showing the typical range.
# Outliers: Points beyond the whiskers (unusual values).
# Insight
# Compare medians: Are 3-bedroom houses pricier than 2-bedroom ones?
# Check spread: Do prices vary more for some bedroom counts?
# Spot outliers: Are there unusually expensive 1-bedroom homes?
# Key Point
# Boxplots excel at comparing distributions and highlighting variability and outliers across groups.