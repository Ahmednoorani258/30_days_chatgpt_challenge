def binary_search(lst, target):
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2  # Find the middle index
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example
sorted_list = [1, 2, 3, 4, 5, 6, 7]
target = 4
index = binary_search(sorted_list, target)
print(f"Index of {target}: {index}")

# Explanation:

# Binary search works on a sorted list by repeatedly dividing the search range in half.
# Initialize left and right pointers to the start and end of the list.
# Calculate the middle index mid. If lst[mid] is the target, return its index.
# If lst[mid] is less than the target, search the right half (left = mid + 1).
# If lst[mid] is greater, search the left half (right = mid - 1).
# If the pointers cross (left > right), the target isn’t in the list, so return -1.
# Time complexity: O(log n), where n is the length of the list.