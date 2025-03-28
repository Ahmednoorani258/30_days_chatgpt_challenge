# 📌 2. Sorting Algorithms

"""
Sorting algorithms rearrange a list into a specific order (e.g., ascending). This file covers both simple and efficient sorting algorithms.

---

## Comparison of Sorting Algorithms:

| Algorithm       | Time Complexity (Best) | Time Complexity (Average) | Time Complexity (Worst) | Space Complexity | Stable |
|------------------|-------------------------|----------------------------|--------------------------|------------------|--------|
| Bubble Sort      | O(n)                   | O(n²)                     | O(n²)                   | O(1)             | Yes    |
| Selection Sort   | O(n²)                  | O(n²)                     | O(n²)                   | O(1)             | No     |
| Insertion Sort   | O(n)                   | O(n²)                     | O(n²)                   | O(1)             | Yes    |
| Merge Sort       | O(n log n)             | O(n log n)                | O(n log n)              | O(n)             | Yes    |
| Quick Sort       | O(n log n)             | O(n log n)                | O(n²)                   | O(log n)         | No     |

---

### Key Notes:
1. **Bubble Sort**:
   - Simple but inefficient for large datasets.
   - Stable because it preserves the relative order of equal elements.

2. **Selection Sort**:
   - Always performs O(n²) comparisons, regardless of input.
   - Not stable because swapping can change the relative order of equal elements.

3. **Insertion Sort**:
   - Efficient for small or nearly sorted datasets.
   - Stable because it inserts elements in their correct position without unnecessary swaps.

4. **Merge Sort**:
   - Divide-and-conquer algorithm with consistent O(n log n) performance.
   - Requires additional space for merging, making it less space-efficient.

5. **Quick Sort**:
   - Very efficient on average but can degrade to O(n²) if the pivot is poorly chosen.
   - Not stable because partitioning can change the relative order of equal elements.

"""