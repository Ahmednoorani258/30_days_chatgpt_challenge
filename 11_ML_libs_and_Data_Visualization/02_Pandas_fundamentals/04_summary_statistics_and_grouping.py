import pandas as pd

# Create a DataFrame
df = pd.read_csv('house_prices.csv')




# Summary stats (mean, median, etc.)
print("Summary:\n", df.describe())

# Group by 'Bedrooms' and compute mean
grouped = df.groupby('Bedrooms').mean()
print("Grouped by Bedrooms:\n", grouped)


# What’s Happening?:
# describe(): Gives count, mean, std, min, max, etc., for numerical columns.
# groupby(): Aggregates data by a category (e.g., average price per bedroom count).
# Why?: Reveals patterns (e.g., do more bedrooms increase price?).