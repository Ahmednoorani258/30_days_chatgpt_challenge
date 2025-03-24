def remove_duplicates(lst):
    seen = set()  # Track seen elements
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Example
numbers = [1, 2, 3, 2, 4, 1, 5]
unique_list = remove_duplicates(numbers)
print(f"List without duplicates: {unique_list}")

# Explanation:

# Use a set to keep track of elements we’ve already seen.
# Iterate through the list and only add an element to result if it hasn’t been seen before.
# This preserves the original order while removing duplicates.
# Time complexity: O(n), where n is the length of the list.