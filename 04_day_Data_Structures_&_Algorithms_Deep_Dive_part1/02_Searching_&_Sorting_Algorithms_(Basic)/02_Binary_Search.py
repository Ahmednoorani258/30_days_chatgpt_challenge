# 2. Binary Search

# What It Does

# Binary Search is an efficient algorithm to find an element in a sorted list. It works by repeatedly dividing the search range in half, making it much faster than Linear Search for large datasets.

# How It Works

# Start with the entire sorted list.
# Find the middle element.
# Compare it with the target:
# If it matches, return the index.
# If the target is smaller, search the left half.
# If the target is larger, search the right half.
# Repeat until the target is found or the search range is empty.

# Details

# Input: Requires a sorted list.
# Time Complexity:
# Worst-case: O(log n) – search space halves each step.
# Best-case: O(1) – target is the middle element.
# Space Complexity: O(1) – uses constant extra space.

# Advantages:

# Very fast for large sorted lists.

# Disadvantages:

# List must be sorted first, which takes extra time if not already done.
# Use Case
# Ideal for large, static, sorted datasets like database lookups.


def binary_search(arr, target):
    # assign left and right pointers
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


sorted_list = [1, 2, 3, 4, 5, 7, 9]
target = 5
result = binary_search(sorted_list, target)

print(f"Sorted List: {sorted_list}")
print(f"Target: {target}")
if result != -1:
    print(f"Element found at index: {result}")
else:
    print("Element not found")
