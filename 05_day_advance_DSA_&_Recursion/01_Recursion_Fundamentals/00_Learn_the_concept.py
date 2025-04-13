# 📌 1️⃣ Recursion Fundamentals

"""
Recursion is a powerful problem-solving technique where a function calls itself to break down a problem
into smaller, manageable subproblems. To master recursion, you need to understand two key components:

1. **Base Case**:
   - The condition that stops the recursion. Without it, the function would call itself indefinitely
     (infinite recursion), leading to a stack overflow.

2. **Recursive Case**:
   - The part of the function where it calls itself with a modified input, moving closer to the base case.

Recursion is like solving a puzzle by reducing it to simpler versions of the same puzzle until it’s trivial to solve.
"""

# Common Recursive Algorithms
# Let’s explore three classic examples: factorial calculation, the Fibonacci sequence, and binary search.

# ----------------------------------------------
# Example 1: Factorial Calculation
# ----------------------------------------------

"""
The factorial of a number `n`, denoted as `n!`, is the product of all positive integers up to `n`.

For example:
5! = 5 × 4 × 3 × 2 × 1 = 120

**Base Case**: If `n == 0` or `n == 1`, return 1.
**Recursive Case**: `n! = n × (n - 1)!`
"""


def factorial(n):
    # Base Case
    if n == 0 or n == 1:
        return 1
    # Recursive Case
    return n * factorial(n - 1)


# Example usage
print("Factorial of 5:", factorial(5))  # Output: 120


# How it works for factorial(5):

"""
The recursive process for calculating factorial(5) unfolds as follows:

1. factorial(5) = 5 × factorial(4)
2. factorial(4) = 4 × factorial(3)
3. factorial(3) = 3 × factorial(2)
4. factorial(2) = 2 × factorial(1)
5. factorial(1) = 1 (base case)

Now, the recursion starts to unwind:

1. factorial(2) = 2 × 1 = 2
2. factorial(3) = 3 × 2 = 6
3. factorial(4) = 4 × 6 = 24
4. factorial(5) = 5 × 24 = 120

Final result: factorial(5) = 120
"""


# ----------------------------------------------
# Example 2: Fibonacci Sequence
# ----------------------------------------------

"""
The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones.

For example:
F(0) = 0, F(1) = 1, F(2) = 1, F(3) = 2, F(4) = 3, F(5) = 5, ...

**Base Case**: If `n == 0`, return 0. If `n == 1`, return 1.
**Recursive Case**: F(n) = F(n - 1) + F(n - 2)
"""


def fibonacci(n):
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Example usage
print("Fibonacci of 5:", fibonacci(5))  # Output: 5

"""
Note: This recursive approach recalculates values multiple times (e.g., `fibonacci(3)` is computed repeatedly), 
making it inefficient for large `n` due to its exponential time complexity O(2^n). 

For optimization, consider using techniques like:
1. **Memoization**: Store the results of previous calculations to avoid redundant computations.
2. **Iteration**: Use an iterative approach to compute the Fibonacci sequence more efficiently.

While this recursive form is not optimal for large inputs, it is excellent for understanding the concept of recursion.
"""
# ----------------------------------------------
# Example 3: Binary Search (Recursive)
# ----------------------------------------------

"""
Binary Search is an efficient algorithm for finding an element in a sorted list by repeatedly dividing the search range in half.

**Base Case**: If the target is found or the search range is empty.
**Recursive Case**: Narrow down the search range based on the comparison with the middle element.
"""


def binary_search(arr, target, left, right):
    # Base Case: If the range is invalid
    if left > right:
        return -1  # Target not found

    mid = (left + right) // 2

    # Base Case: If the target is found
    if arr[mid] == target:
        return mid

    # Recursive Case: Search in the left or right half
    if target < arr[mid]:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)


# Example usage
sorted_list = [1, 3, 5, 7, 9, 11]
target = 7
result = binary_search(sorted_list, target, 0, len(sorted_list) - 1)
if result != -1:
    print(
        f"Target {target} found at index {result}"
    )  # Output: Target 7 found at index 3
else:
    print(f"Target {target} not found")

"""
How it works for finding 10 using Binary Search:

1. Start with the sorted list: [2, 3, 4, 10, 40, 50].
2. Calculate the middle index: mid = 2 (value = 4).
   - Compare the target (10) with the middle value (4).
   - Since 10 > 4, search in the right half of the list.

3. The new search range is: [10, 40, 50].
   - Calculate the middle index: mid = 0 (value = 10).
   - Compare the target (10) with the middle value (10).
   - Target found!

**Time Complexity**: O(log n)
- At each step, the search range is halved, making the algorithm very efficient for large datasets.
"""
