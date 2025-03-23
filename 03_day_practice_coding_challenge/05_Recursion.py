# Question: Fibonacci Sequence Using Recursion

# Write a recursive function fibonacci(n) that takes a non-negative integer n as input and returns the n-th number in the Fibonacci sequence. The Fibonacci sequence is defined as follows:

# The 0th Fibonacci number is 0 (F(0) = 0).
# The 1st Fibonacci number is 1 (F(1) = 1).
# For n ≥ 2, the n-th Fibonacci number is the sum of the two preceding numbers: F(n) = F(n-1) + F(n-2).

# Requirements:
# Use recursion to compute the Fibonacci number (no loops allowed).
# Handle edge cases:
# If n is negative, return None to indicate an invalid input.
# Assume n is an integer.

# Examples:
# fibonacci(0) → 0 (0th Fibonacci number)
# fibonacci(1) → 1 (1st Fibonacci number)
# fibonacci(5) → 5 (since the sequence is 0, 1, 1, 2, 3, 5)
# fibonacci(7) → 13 (sequence: 0, 1, 1, 2, 3, 5, 8, 13)
# fibonacci(-1) → None (negative input is invalid)

def fibonacci1(n):
    if n < 0:
        return None
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci1(n - 1) + fibonacci1(n - 2)


def fibonacci2(n):
    """Return the n-th Fibonacci number using recursion with a helper parameter."""
    if n < 0:
        return None
    def fib_helper(pos):
        if pos == 0:
            return 0
        if pos == 1:
            return 1
        return fib_helper(pos - 1) + fib_helper(pos - 2)
    return fib_helper(n)

def fibonacci3(n):
    """Return the n-th Fibonacci number using tail recursion."""
    if n < 0:
        return None
    def fib_tail(pos, prev=0, curr=1):
        if pos == 0:
            return prev
        if pos == 1:
            return curr
        return fib_tail(pos - 1, curr, prev + curr)
    return fib_tail(n)


# Comparison of Approaches
# Differences
# Approach 1 (Basic): Direct recursion matching the definition F(n) = F(n-1) + F(n-2).
# Approach 2 (Helper): Wraps the recursion in a helper function for clarity, same logic.
# Approach 3 (Tail): Uses tail recursion with accumulated values, avoiding redundant sums.
# Efficiency
# Basic & Helper: O(2^n) time, O(n) space – inefficient due to exponential branching.
# Tail: O(n) time, O(n) space in Python (O(1) with TCO) – much faster but still limited by Python’s stack.
# Readability
# Basic: Most intuitive, directly follows the mathematical definition.
# Helper: Slightly more structured with a nested function.
# Tail: Requires understanding accumulators but is elegant once grasped.
# Correctness
# All pass the test cases: 0, 1, 5, 13, None for inputs 0, 1, 5, 7, -1.
# Optimization Note
# The tail-recursive approach is the most efficient recursively, but for large n, all suffer from Python’s recursion depth limit (~1000). A true optimization would use iteration or memoization, but since recursion is required, Approach 3 is the best of these.

# Test Cases
print(fibonacci1(0) )#→ 0
print(fibonacci2(1) )#→ 1
print(fibonacci3(5) )#→ 5
print(fibonacci2(7) )#→ 13
print(fibonacci3(-1))# → None