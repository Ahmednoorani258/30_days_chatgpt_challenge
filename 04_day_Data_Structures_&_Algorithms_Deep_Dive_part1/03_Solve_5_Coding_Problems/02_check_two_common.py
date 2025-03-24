def have_common_elements(lst1, lst2):
    set1 = set(lst1)  # Convert first list to a set
    for item in lst2:
        if item in set1:
            return True
    return False

# Example
list1 = [1, 2, 3, 4]
list2 = [4, 5, 6]
result = have_common_elements(list1, list2)
print(f"Have common elements? {result}")

# Explanation:

# Convert lst1 to a set for O(1) lookup time.
# Iterate through lst2 and check if any element exists in set1.
# If a match is found, return True. If no match is found after checking all elements, return False.
# Time complexity: O(n + m), where n and m are the lengths of the two lists. Using a set makes lookups efficient.