def merge_sorted_lists(list1, list2):
    merged = []
    i, j = 0, 0  # Pointers for list1 and list2
    
    # Compare elements and merge
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements from list1, if any
    merged.extend(list1[i:])
    # Add remaining elements from list2, if any
    merged.extend(list2[j:])
    
    return merged

# Example
list1 = [1, 3, 5]
list2 = [2, 4, 6]
result = merge_sorted_lists(list1, list2)
print(f"Merged sorted list: {result}")


# Explanation:

# Use two pointers i and j to track positions in list1 and list2.
# Compare elements at the current positions and append the smaller one to merged.
# Move the pointer of the list from which we took the element.
# Once one list is exhausted, append the remaining elements of the other list.
# Time complexity: O(n + m), where n and m are the lengths of the two lists. Space complexity: O(n + m) for the output list.