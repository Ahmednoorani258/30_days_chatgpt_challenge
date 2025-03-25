# 📌 1. Sum of a List Recursively
"""
This program calculates the sum of a list of numbers using recursion.

### Key Points:
1. **Base Case:** If the list is empty, return 0.
2. **Recursive Case:** Add the first element of the list to the sum of the rest of the list.

### Example:
For the list [1, 2, 3, 4]:
- Sum = 1 + (2 + (3 + (4 + 0))) = 10
"""

def recursive_sum(lst):
    # Base Case: If the list is empty, return 0
    if not lst:
        return 0
    # Recursive Case: Add the first element to the sum of the rest of the list
    
    return lst[0] + recursive_sum(lst[1:])

# Example usage
numbers = [1, 2, 3, 4]
result = recursive_sum(numbers)
print(f"The sum of {numbers} is: {result}")  # Output: The sum of [1, 2, 3, 4] is: 10