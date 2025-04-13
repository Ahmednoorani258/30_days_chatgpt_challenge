# Question : Two Sum
# Problem:

# Write a function two_sum(numbers, target) that takes a list of integers numbers and an integer target as input. The function should return a list containing the indices of two numbers in numbers that add up to target. You may assume that each input has exactly one solution, and you cannot use the same element twice (i.e., the two indices must be different). If no solution exists, return an empty list.

# Requirements:
# Return the indices in any order.
# Use a solution with less than O(n²) time complexity if possible.

# Examples:
# two_sum([2, 7, 11, 15], 9) → [0, 1] (because numbers[0] + numbers[1] = 2 + 7 = 9)
# two_sum([3, 2, 4], 6) → [1, 2] (because numbers[1] + numbers[2] = 2 + 4 = 6)
# two_sum([3, 3], 6) → [0, 1] (because numbers[0] + numbers[1] = 3 + 3 = 6)
# two_sum([1, 2, 3], 10) → [] (no pair sums to 10)

# Plan:
# 1. Create a dictionary to store the index of the numbers.
# 2. Iterate through the list and find the difference between the target and the current number.
# 3. If the difference is in the dictionary, return the index of the current number and the index of the difference.
# 4. If the difference is not in the dictionary, add the current number and its index to the dictionary.
# 5. If no solution is found, return an empty list.


def two_sum(numbers, target):
    """Find indices of two numbers in a list that add up to the target."""
    # Dictionary to store number-to-index mappings
    seen_numbers = {}

    # Iterate through the list with index and value
    for index, current_num in enumerate(numbers):
        # Calculate the number needed to reach the target
        complement = target - current_num

        # If the complement exists in the dictionary, we found a pair
        if complement in seen_numbers:
            return [seen_numbers[complement], index]

        # Otherwise, add the current number and its index
        seen_numbers[current_num] = index

    # No pair found after checking all numbers
    return []


# Test cases
print(two_sum([2, 7, 11, 15], 18))  # [1, 2]
print(two_sum([3, 2, 4], 6))  # [1, 2]
print(two_sum([3, 3], 6))  # [0, 1]
print(two_sum([1, 2, 3], 10))  # []

# Second Approach:


def two_sum(numbers, target):
    """Find two numbers that add to target using nested loops."""
    # Check each pair of numbers
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return [i, j]
    return []  # No pair found


# Pros: Extremely clear—no data structures, just basic loops and addition.
# Cons: O(n²) time, doesn’t meet the efficiency preference.
