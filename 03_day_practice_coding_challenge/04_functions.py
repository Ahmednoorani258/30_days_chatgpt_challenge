# Question: Factorial Function

# Write a function factorial(n) that takes a non-negative integer n as input and returns its factorial. The factorial of a number n (denoted as n!) is the product of all positive integers less than or equal to n. For example, 5! = 5 × 4 × 3 × 2 × 1 = 120. By definition, 0! = 1.

# Requirements:
# The function must use a loop (iterative approach) to calculate the factorial, not recursion.
# Handle edge cases:
# If n is negative, return None to indicate an invalid input.
# If n is 0, return 1.

# Assume n is an integer.


# Examples:
# factorial(5) → 120 (since 5! = 5 × 4 × 3 × 2 × 1 = 120)
# factorial(0) → 1 (since 0! = 1 by definition)
# factorial(3) → 6 (since 3! = 3 × 2 × 1 = 6)
# factorial(-1) → None (negative numbers don’t have factorials in this context)


# Test Cases:

# print(factorial(5))   # Should output: 120
# print(factorial(0))   # Should output: 1
# print(factorial(3))   # Should output: 6
# print(factorial(-1))  # Should output: None
# This question focuses on:


def factorial(n):
    if n < 0:
        return None
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result


# Test the function with example inputs
print(factorial(5))  # 120
