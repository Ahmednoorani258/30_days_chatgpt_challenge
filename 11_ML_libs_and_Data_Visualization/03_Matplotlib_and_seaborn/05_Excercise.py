# Exercise: Applying Skills to the Titanic Dataset
# Let’s apply these skills to the Titanic dataset (assumed loaded as df_titanic, with columns Age, Fare, and Pclass). Complete these tasks:

# Plot a histogram of Age to see its distribution.
# Create a scatter plot of Age vs. Fare to explore relationships.
# Use a boxplot to compare Fare across Pclass categories.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Plot a histogram of Age to see its distribution.

df =pd.read_csv('Titanic-Dataset.csv')

plt.hist(df['Age'], bins=20,color = 'green')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Expected Insight: The histogram might show a peak in young adults (20-30 years) with fewer children and elderly passengers.

# _________________________________________
# _________________________________________

# Create a scatter plot of Age vs. Fare to explore relationships.


sns.scatterplot(x='Age', y='Fare', data=df)
plt.title('Age vs. Fare')
plt.show()

# Expected Insight: The scatter plot might show a positive correlation between age and fare, with older passengers paying higher fares.

# _________________________________________
# _________________________________________

# Use a boxplot to compare Fare across Pclass categories.

sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.show()

# Expected Insight: First-class (Pclass=1) likely has higher fares and more variability, with outliers indicating extravagant tickets.


# Summary
# You’ve now mastered the essentials of data visualization with Matplotlib and Seaborn:

# Histograms: Reveal how a variable is distributed.
# Scatter Plots: Explore relationships between two variables.
# Boxplots: Compare distributions across categories.
# Customization: Enhance readability and appeal with labels, colors, and sizes.