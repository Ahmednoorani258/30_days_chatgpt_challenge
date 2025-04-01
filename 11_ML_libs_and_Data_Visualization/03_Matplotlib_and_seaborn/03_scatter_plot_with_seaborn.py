# Scatter Plot with Seaborn
# A scatter plot visualizes the relationship between two continuous variables, such as house size and price.

# Step 1: Importing Seaborn
# Seaborn builds on Matplotlib, offering a simpler syntax and more attractive defaults. Import it with:

# run this command to install: pip install seaborn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Step 2: Plotting a Scatter Plot
# Use sns.scatterplot() to plot Size against Price:

df = pd.read_csv('Housing.csv')
# Scatter plot: Size vs Price
sns.scatterplot(x='area', y='price', data=df)
plt.title('House Size vs Price')
plt.show()


# What’s Happening?
# sns.scatterplot(x='Size', y='Price', data=df'):
# x='Size': Plots house size on the x-axis.
# y='Price': Plots price on the y-axis.
# data=df: Specifies the DataFrame containing the data.
# plt.title('House Size vs Price'): Adds a title.
# plt.show(): Displays the plot.
# Insight
# The scatter plot reveals relationships:

# Do larger houses tend to cost more (a positive trend)?
# Is there no clear pattern (random scatter)?
# Are there outliers (e.g., a small house with a huge price)?
# Key Point
# Scatter plots are essential for exploring correlations and trends between two variables.