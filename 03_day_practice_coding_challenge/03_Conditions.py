# Question: Check for Prime Numbers Using Conditionals


# Write a function is_prime(n) that takes an integer n greater than 0 as input and returns True if n is a prime number, and False otherwise. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Use conditionals to handle edge cases and optimize the solution by checking divisibility only up to the square root of n.

# Requirements:
# Handle edge cases such as n = 1 (not prime) and n = 2 (prime).
# Use conditionals (if statements) to check divisibility and determine the result.
# Optimize the function by limiting the divisibility check to numbers up to the square root of n.
# Examples:
# is_prime(2) → True (2 is prime)
# is_prime(4) → False (4 is divisible by 2)
# is_prime(17) → True (17 has no divisors other than 1 and itself)
# is_prime(1) → False (1 is not prime)
# This question tests your understanding of conditionals, loops, and basic optimization techniques while keeping the focus on determining whether a number is prime. You can try solving it in your preferred programming language!


# First approach


def is_prime(n):
    if n <= 1:  # if n is less than or equal to 1, it is not prime
        return False
    if n <= 3:  # if n is less than or equal to 3, it is prime
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    # loop through the numbers from 5 to the square root of n
    while i * i <= n:
        # check if n is divisible by i or i + 2
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# Test the function with example inputs
print(is_prime(2))  # True
print(is_prime(4))  # False
print(is_prime(17))  # True
print(is_prime(1))  # False
print(is_prime(1009))  # True
print(is_prime(1013))  # True


# _____________________________________________________________
# Second approach
# _____________________________________________________________


def is_prime(n):
    # If n is less than 2, it's not prime (covers 1, 0, and negative numbers)
    if n < 2:
        return False
    # Check for divisors from 2 up to the square root of n
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False  # Found a divisor, so n is not prime
    return True  # No divisors found, so n is prime
