# 7. Exercise
# Let’s put your skills to the test:

# Create a NumPy array of 10 random integers between 1 and 100 using np.random.randint(1, 101, 10).
# Compute its sum, mean, and maximum value (use np.max()).
# Extract the first 4 elements and all elements greater than 50.


import numpy as np

# Step 1: Create the array
ex_arr = np.random.randint(1, 101, 10)
print("Array:", ex_arr)
# e.g., [34 67 12 89 45 23 78 91 56 19]

# Step 2: Compute sum, mean, and max
print("Sum:", np.sum(ex_arr))  # e.g., 514
print("Mean:", np.mean(ex_arr))  # e.g., 51.4
print("Max:", np.max(ex_arr))  # e.g., 91

# Step 3: Slicing and boolean indexing
print("First four elements:", ex_arr[:4])  # e.g., [34 67 12 89]
print("Elements > 50:", ex_arr[ex_arr > 50])  # e.g., [67 89 78 91 56]