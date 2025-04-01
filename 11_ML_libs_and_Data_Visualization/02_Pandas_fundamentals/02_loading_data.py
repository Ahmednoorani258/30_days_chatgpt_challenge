# Loading a Dataset
# Before analyzing data, you need to load it into a DataFrame.

# __________________________________________________________________
# Step 1: Importing Pandas
# __________________________________________________________________
# Start by importing the Pandas library. The convention is to alias it as pd:

import pandas as pd

# __________________________________________________________________
# Loading a CSV File
# __________________________________________________________________

# Load the CSV file (replace 'house_prices.csv' with your file path)
df = pd.read_csv('house_prices.csv')

# Preview the first 5 rows
print(df.head())

# What’s Happening?: pd.read_csv() loads the file into a DataFrame. df.head() shows the top 5 rows (e.g., Price, Size, Bedrooms).
# Key Point: DataFrames are ideal for tabular data.


