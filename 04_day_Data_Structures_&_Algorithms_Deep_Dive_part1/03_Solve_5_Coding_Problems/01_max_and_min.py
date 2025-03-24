def find_max_min(lst):
    if not lst:  # Check if the list is empty
        return None, None
    # assign the first element to max_val and min_val
    # to avoid the case where max_val and min_val are assigned to 0
    max_val = min_val = lst[0]  # Start with the first element
    for num in lst:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num
    return max_val, min_val

# Example
numbers = [4, 2, 7, 1, 9, 3]
max_num, min_num = find_max_min(numbers)
print(f"Maximum: {max_num}, Minimum: {min_num}")


# Explanation:

# We define a function find_max_min that takes a list lst as input.
# If the list is empty, return None for both max and min.
# Initialize max_val and min_val with the first element of the list.
# Iterate through the list, updating max_val if we find a larger number and min_val if we find a smaller number.
# Return the pair of values. This has a time complexity of O(n), where n is the length of the list.