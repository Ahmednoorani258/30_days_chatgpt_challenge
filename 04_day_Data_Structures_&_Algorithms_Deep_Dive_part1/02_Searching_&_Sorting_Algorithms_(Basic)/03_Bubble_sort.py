# Bubble Sort

# Bubble Sort is a basic sorting algorithm that arranges a list in order (e.g., ascending). It repeatedly compares adjacent elements and swaps them if they’re in the wrong order, “bubbling” larger elements to the end.

# How It Works

# Start at the beginning of the list.
# Compare each pair of adjacent elements.
# Swap them if the first is greater than the second.
# Repeat this process for the entire list, reducing the range each time as the largest elements settle at the end.

# Details

# Input: Works on any list (sorted or unsorted).
# Time Complexity:
# Worst-case: O(n²) – list is in reverse order.
# Best-case: O(n) – list is already sorted (with optimization).
# Space Complexity: O(1) – sorts in place.

# Advantages:

# Simple to understand and code.

# Disadvantages:

# Very slow for large lists due to O(n²) complexity.

# Use Case

# Great for learning sorting concepts or sorting tiny lists, but not practical for real-world large datasets.


def bubble_sort(arr):
    # assign length
    n = len(arr)
    # Traverse through all elements in the list
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the list from 0 to n-i-1
            if arr[j] > arr[j + 1]:
            # Swap if the element found is greater than the next element
                # Swap elements
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap elements
    return arr

# Example usage
unsorted_list = [64, 34, 25, 12, 22, 11, 90]
print(f"Unsorted List: {unsorted_list}")
sorted_list = bubble_sort(unsorted_list)
print(f"Sorted List: {sorted_list}")