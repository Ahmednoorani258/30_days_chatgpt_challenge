import pandas as pd

# Create a DataFrame
df = pd.read_csv("house_prices.csv")

# Display the first few rows
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing 'Price' values with the mean
df["Price"].fillna(df["Price"].mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print("After cleaning:\n", df.head())


# What’s Happening?:
# isnull().sum(): Counts missing values per column.
# fillna(): Replaces NaN with the column mean.
# drop_duplicates(): Removes identical rows.
# Key Point: Cleaning ensures data quality for analysis.
