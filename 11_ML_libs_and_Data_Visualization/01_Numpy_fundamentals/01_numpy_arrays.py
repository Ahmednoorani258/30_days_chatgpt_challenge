# To use NumPy, you first need to import it. The standard convention is to import it as np:
import numpy as np


# Creating a 1D Array
# A one-dimensional (1D) array is like a list of elements. Here’s an example:
# __________________________________________________________________
# __________________________________________________________________

# Create a 1D array of 5 random numbers between 0 and 1
arr = np.random.rand(5)
print("Array:", arr)


# What’s Happening?:
#
# np.random.rand(5) generates an array of 5 random floating-point numbers between 0 and 1. Your output might look like [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864].

# Key Point:
#
# Arrays are homogeneous, meaning all elements must be of the same data type (e.g., all floats). This uniformity enables fast computations.


# __________________________________________________________________
# __________________________________________________________________
# Creating a 2D Array
# NumPy supports multi-dimensional arrays. A 2D array resembles a matrix with rows and columns:

# Create a 2D array (3 rows, 4 columns) of random integers between 1 and 100
arr_2d = np.random.randint(1, 101, size=(3, 4))
print("2D Array:\n", arr_2d)

# What’s Happening?: np.random.randint(1, 101, size=(3, 4)) creates a 2D array with 3 rows and 4 columns, filled with random integers from 1 to 100 (inclusive of 1, exclusive of 101). Your output might look like:

# Key Point: Multi-dimensional arrays are ideal for representing complex data, such as matrices in linear algebra or pixel grids in images.

# __________________________________________________________________
# __________________________________________________________________
# Other Ways to Create Arrays
# NumPy offers several methods to create arrays tailored to specific needs:


# from a python list

list_data = [1, 2, 3, 4]
arr_from_list = np.array(list_data)
print("Array from list:", arr_from_list)  # Output: [1 2 3 4]


# filled with 0s or 1s

zeros_arr = np.zeros(5)  # 1D array of zeros
print("Zeros:", zeros_arr)  # [0. 0. 0. 0. 0.]

ones_arr = np.ones((2, 3))  # 2D array of ones
print("Ones:\n", ones_arr)  # [[1. 1. 1.] [1. 1. 1.]]

# sequence with arange

seq_arr = np.arange(0, 10, 2)  # Start at 0, end before 10, step by 2
print("Sequence:", seq_arr)  # [0 2 4 6 8]

# Evenly Spaced Numbers with linspace

lin_arr = np.linspace(0, 1, 5)  # 5 numbers from 0 to 1, inclusive
print("Linear space:", lin_arr)  # [0.   0.25 0.5  0.75 1.  ]
