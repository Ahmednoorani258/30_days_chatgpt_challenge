import numpy as np

# Slicing and Indexing
# NumPy offers powerful tools to access and manipulate specific parts of arrays.

# ______________________________________________________________________
# Basic Indexing
# ______________________________________________________________________
arr = np.random.rand(5)
# 1D Arrays: Similar to Python lists.

print("First element:", arr[0])  # e.g., 0.37454012
print("Last element:", arr[-1])  # e.g., 0.15601864


arr_2d = np.random.randint(1, 101, size=(3, 4))
print("Element at (1, 2):", arr_2d[1, 2])  # e.g., 56 (second row, third column)


# ______________________________________________________________________
# Slicing
# ______________________________________________________________________

# Slicing extracts a subset of the array using the syntax [start:stop:step].

# 1d array slicing

# First three elements
first_three = arr[:3]
print("First three:", first_three)  # e.g., [0.37454012 0.95071431 0.73199394]

# Last two elements
last_two = arr[-2:]
print("Last two:", last_two)  # e.g., [0.59865848 0.15601864]

# Every other element
every_other = arr[::2]
print("Every other:", every_other)  # e.g., [0.37454012 0.73199394 0.15601864]

# ______________________________________________________________________
# 2d array slicing
# ______________________________________________________________________
#
# First two rows, all columns
first_two_rows = arr_2d[:2, :]
print("First two rows:\n", first_two_rows)
# e.g., [[45 67 23 89]
#        [12 34 56 78]]

# All rows, first three columns
first_three_cols = arr_2d[:, :3]
print("First three columns:\n", first_three_cols)
# e.g., [[45 67 23]
#        [12 34 56]
#        [90 11 32]]


# ______________________________________________________________________
# Boolean Indexing

# Elements greater than 0.5
greater_than_half = arr[arr > 0.5]
print("Greater than 0.5:", greater_than_half)  # e.g., [0.95071431 0.73199394 0.59865848]
# Elements less than the mean
mean_val = np.mean(arr)
less_than_mean = arr[arr < mean_val]
print("Less than mean:", less_than_mean)  # e.g., [0.37454012 0.15601864]

# What’s Happening?: arr > 0.5 creates a boolean array (e.g., [False True True True False]), which is used to select matching elements.

# ______________________________________________________________________
# Fancy Indexing
# ______________________________________________________________________

# Access multiple elements using a list of indices:

# Elements at indices 0, 2, and 4
fancy_indexed = arr[[0, 2, 4]]
print("Fancy indexing:", fancy_indexed)  # e.g., [0.37454012 0.73199394 0.15601864]

# ______________________________________________________________________
# Array Shape and reshapping
# ______________________________________________________________________

# Arrays have a shape attribute that describes their dimensions:

print("Shape of arr:", arr.shape)  # (5,) – 1D array with 5 elements
print("Shape of arr_2d:", arr_2d.shape)  # (3, 4) – 2D array with 3 rows, 4 columns

# You can reshape arrays to change their structure without altering the data:

# Reshape 1D array to 2D (5 rows, 1 column)
reshaped_arr = arr.reshape(5, 1)
print("Reshaped array:\n", reshaped_arr)
# e.g., [[0.37454012]
#        [0.95071431]
#        [0.73199394]
#        [0.59865848]
#        [0.15601864]]

# ______________________________________________________________________
# BroadCasting
# ______________________________________________________________________

# Broadcasting allows NumPy to operate on arrays of different shapes by automatically expanding the smaller array:


# Add a scalar to every element
print("Add 10 to each:", arr + 10)  # e.g., [10.37454012 10.95071431 ...]

# Add a 1D array to a 2D array
row_to_add = np.array([1, 2, 3, 4])
print("Broadcasting addition:\n", arr_2d + row_to_add)
# e.g., [[46 69 26 93]
#        [13 36 59 82]
#        [91 13 35 58]]


# What’s Happening?: row_to_add is “broadcasted” to match the shape of arr_2d, adding each element to the corresponding column.