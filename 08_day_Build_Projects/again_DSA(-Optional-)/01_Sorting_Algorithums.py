# 📌 2. Sorting Algorithms

"""
Sorting algorithms rearrange a list into a specific order (e.g., ascending). This file covers both simple and efficient sorting algorithms.

---

## Simple Sorts (O(n²)):
1. **Bubble Sort**:
   - **How it Works**: Compares adjacent elements and swaps them if they’re in the wrong order. Repeats until no more swaps are needed.
   - **Time Complexity**: O(n²) — requires nested loops.
   - **Space Complexity**: O(1) — in-place sorting.

2. **Selection Sort**:
   - **How it Works**: Repeatedly selects the smallest element from the unsorted portion and places it at the end of the sorted portion.
   - **Time Complexity**: O(n²) — nested loops to find the minimum each time.
   - **Space Complexity**: O(1) — in-place sorting.

3. **Insertion Sort**:
   - **How it Works**: Builds a sorted portion by inserting each new element into its correct position.
   - **Time Complexity**: O(n²) worst case, but O(n) for nearly sorted data.
   - **Space Complexity**: O(1) — in-place sorting.

---

## Efficient Sorts (O(n log n)):
1. **Merge Sort**:
   - **How it Works**: A divide-and-conquer algorithm. Divides the list into halves, recursively sorts them, and merges the sorted halves.
   - **Time Complexity**: O(n log n) — log n levels of division, each requiring O(n) to merge.
   - **Space Complexity**: O(n) — needs extra space for merging.

2. **Quick Sort**:
   - **How it Works**: Picks a pivot, partitions the list (smaller elements left, larger right), and recursively sorts the partitions.
   - **Time Complexity**: O(n log n) average, O(n²) worst case (e.g., poor pivot choice like smallest/largest element).
   - **Space Complexity**: O(log n) — recursion stack.

---
"""

# ----------------------------------------------
# Bubble Sort
# ----------------------------------------------


def bubble_sort(arr):
    """
    Sorts an array using the Bubble Sort algorithm.
    Args:
        arr (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# ----------------------------------------------
# Selection Sort
# ----------------------------------------------


def selection_sort(arr):
    """
    Sorts an array using the Selection Sort algorithm.
    Args:
        arr (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# ----------------------------------------------
# Insertion Sort
# ----------------------------------------------


def insertion_sort(arr):
    """
    Sorts an array using the Insertion Sort algorithm.
    Args:
        arr (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# ----------------------------------------------
# Merge Sort
# ----------------------------------------------


def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    Args:
        arr (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    """
    Merges two sorted lists into one sorted list.
    Args:
        left (list): The first sorted list.
        right (list): The second sorted list.
    Returns:
        list: The merged sorted list.
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ----------------------------------------------
# Quick Sort
# ----------------------------------------------


def quick_sort(arr, low, high):
    """
    Sorts an array using the Quick Sort algorithm.
    Args:
        arr (list): The list to be sorted.
        low (int): The starting index of the list.
        high (int): The ending index of the list.
    Returns:
        list: The sorted list.
    """
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr


def partition(arr, low, high):
    """
    Partitions the array around a pivot for Quick Sort.
    Args:
        arr (list): The list to be partitioned.
        low (int): The starting index.
        high (int): The ending index.
    Returns:
        int: The index of the pivot.
    """
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
