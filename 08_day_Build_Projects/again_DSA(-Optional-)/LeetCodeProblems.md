# 🚀 Intermediate Problem-Solving Guide

This repository contains solutions to 12 intermediate-level problems that are commonly encountered in coding interviews and competitive programming. Each problem is solved using both brute-force and optimized approaches, with detailed explanations and complexity analysis.

---

## 📌 Problems and Solutions

### 1. Two Sum Problem
**Problem Statement**: Find two indices in an array such that their sum equals a given target.

- **Brute Force Approach**: O(n²)
  - Nested loops to check all pairs.
- **Optimized Approach**: O(n) using a Hash Map.
  - Store complements in a dictionary for fast lookup.

---

### 2. Longest Substring Without Repeating Characters
**Problem Statement**: Find the length of the longest substring without repeating characters.

- **Approach**: Sliding Window (O(n)).
  - Use two pointers and a set to track characters in the current window.

---

### 3. Climbing Stairs
**Problem Statement**: Find the number of ways to climb `n` steps, taking 1 or 2 steps at a time.

- **Recursive with Memoization**: O(n).
  - Use a dictionary to store results of subproblems.
- **Iterative DP**: O(n).
  - Use a DP array to calculate the result iteratively.

---

### 4. Finding Longest Consecutive Sequence
**Problem Statement**: Find the length of the longest consecutive sequence in an unsorted array.

- **Approach**: HashSet (O(n)).
  - Use a set for fast lookups and check for the start of a sequence.

---

### 5. Subsets (Power Set)
**Problem Statement**: Generate all possible subsets of a given array.

- **Approach**: Recursive Backtracking (O(2ⁿ)).
  - Use recursion to include/exclude elements and generate subsets.

---

### 6. Product of Array Except Self
**Problem Statement**: Return an array where each element is the product of all other elements, without using division.

- **Approach**: Prefix and Suffix Products (O(n)).
  - Calculate prefix and suffix products in two passes.

---

### 7. Valid Parentheses
**Problem Statement**: Check if a string containing parentheses is valid.

- **Approach**: Stack-Based (O(n)).
  - Use a stack to match opening and closing brackets.

---

### 8. Merge Intervals
**Problem Statement**: Merge overlapping intervals in a list.

- **Approach**: Sorting + Greedy (O(n log n)).
  - Sort intervals and merge overlapping ones.

---

### 9. Kth Largest Element in an Array
**Problem Statement**: Find the k-th largest element in an array.

- **Approach**: Heap (O(n log k)).
  - Use a min-heap to maintain the k largest elements.

---

### 10. Binary Search
**Problem Statement**: Find the index of a target element in a sorted array.

- **Approach**: Binary Search (O(log n)).
  - Use a divide-and-conquer approach to halve the search space.

---

### 11. Find Peak Element
**Problem Statement**: Find a peak element in an array (an element greater than its neighbors).

- **Approach**: Binary Search (O(log n)).
  - Use binary search to find a peak efficiently.

---

## 🛠 Techniques Covered
1. **Sliding Window**: Efficiently handle substrings or subarrays.
2. **Hash Maps**: Fast lookups for complements or unique elements.
3. **Dynamic Programming**: Solve problems by breaking them into subproblems.
4. **Greedy Algorithms**: Make optimal choices at each step.
5. **Binary Search**: Efficiently search in sorted data.
6. **Backtracking**: Explore all possibilities recursively.

---

## 📊 Complexity Analysis
Each problem includes:
- **Time Complexity**: How the runtime scales with input size.
- **Space Complexity**: Additional memory used by the algorithm.

---

## 🚀 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/intermediate-problems.git