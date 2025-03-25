# 📌 Flatten a Nested List
"""
This program flattens a nested list into a single list using recursion.

### Key Points:
1. **Base Case:** If the current item is not a list, append it to the result.
2. **Recursive Case:** If the current item is a list, recursively flatten it.

### Example:
Input: [1, [2, 3], [4, [5, 6]]]
Output: [1, 2, 3, 4, 5, 6]
"""

def flatten_list(nested_list):
    # Initialize an empty list to store the flattened result
    flattened = []
    
    for item in nested_list:
        # Base Case: If the item is not a list, append it to the result
        if not isinstance(item, list):
            flattened.append(item)
        else:
            # Recursive Case: If the item is a list, recursively flatten it
            flattened.extend(flatten_list(item))
    
    return flattened

# Example usage
nested_list = [1, [2, 3], [4, [5, 6]]]
result = flatten_list(nested_list)
print(f"Flattened list: {result}")  # Output: Flattened list: [1, 2, 3, 4, 5, 6]