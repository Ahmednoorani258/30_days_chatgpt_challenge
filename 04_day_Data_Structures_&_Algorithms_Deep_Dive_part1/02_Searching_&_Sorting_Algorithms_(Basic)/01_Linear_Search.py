# 1. Linear Search

# What It Does

# Linear Search is a simple algorithm used to find an element in a list. It checks each element one by one, from the start to the end, until it finds the target element or determines it’s not present.

# How It Works

# Start at the first element.
# Compare it with the target.
# If it matches, return the index.
# If not, move to the next element and repeat until the target is found or the list ends.

# Details

# Input: Works on any list (sorted or unsorted).

# Time Complexity:

# Worst-case: O(n) – target is at the end or not present.
# Best-case: O(1) – target is at the start.
# Space Complexity: O(1) – uses only a few variables.

# Advantages:

# Easy to implement.
# No need for a sorted list.

# Disadvantages:

# Slow for large lists since it checks every element.
# Use Case
# Best for small lists or unsorted data where simplicity is more important than speed.


def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Return index if found
    return -1  # Return -1 if not found


# Example usage
my_list = [4, 2, 7, 1, 9, 3]
target = 7
result = linear_search(my_list, target)

print(f"List: {my_list}")
print(f"Target: {target}")
if result != -1:
    print(f"Element found at index: {result}")
else:
    print("Element not found")
