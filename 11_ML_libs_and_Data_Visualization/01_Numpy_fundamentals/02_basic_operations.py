import numpy as np
# Basic Operations on Arrays
# NumPy’s strength lies in its ability to perform operations on entire arrays efficiently, thanks to vectorization. Let’s explore some fundamental operations.

# ______________________________________________________________________
# A. Element-wise Operations
# ______________________________________________________________________
# You can apply arithmetic operations directly to arrays, and they’ll be performed element-by-element:

# Create two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
print("Addition:", a + b)  # [5 7 9]

# Element-wise multiplication
print("Multiplication:", a * b)  # [4 10 18]

# Element-wise division
print("Division:", b / a)  # [4.  2.5 2. ]

# What’s Happening?: Each element in a is paired with the corresponding element in b for the operation.
# Key Point: This is much faster than looping over lists in pure Python because NumPy operations are implemented in C.


# ______________________________________________________________________
# B. Universal Functions (ufuncs)
# ______________________________________________________________________
# NumPy provides mathematical functions that operate element-wise, called universal functions:

# Square root of each element
print("Square root:", np.sqrt(a))  # [1.         1.41421356 1.73205081]

# Exponential of each element
print("Exponential:", np.exp(a))  # [ 2.71828183  7.3890561  20.08553692]

# ______________________________________________________________________
# C. Aggregation Functions
# ______________________________________________________________________
# These summarize the data in an array:

arr = np.random.rand(5)

# Using our random 1D array from earlier
print("Array:", arr)

# Sum of all elements
total = np.sum(arr)
print("Sum:", total)  # e.g., ~2.81192549

# Mean (average)
mean_val = np.mean(arr)
print("Mean:", mean_val)  # e.g., ~0.5623851

# Standard deviation (spread of data)
std_dev = np.std(arr)
print("Standard Deviation:", std_dev)  # e.g., ~0.27634939

# Maximum and minimum values
max_val = np.max(arr)
min_val = np.min(arr)
print("Max:", max_val, "Min:", min_val)  # e.g., Max: 0.95071431 Min: 0.15601864


# What’s Happening?:
# np.sum(): Adds all elements.
# np.mean(): Computes the average.
# np.std(): Measures variability (how spread out the values are).
# np.max() and np.min(): Find the largest and smallest values.
# Why Vectorization?: These operations process the entire array in one go, avoiding slow Python loops.

