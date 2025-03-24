# 📊 Time Complexity Comparison

This document provides a comparison of the time complexities for **Linear Search**, **Binary Search**, and **Bubble Sort**, along with explanations and key takeaways.

---

## 📌 Time Complexity Table

| **Algorithm**       | **Best Case** | **Average Case** | **Worst Case** | **Space Complexity** |
|----------------------|---------------|------------------|----------------|-----------------------|
| **Linear Search**    | O(1)          | O(n)             | O(n)           | O(1)                 |
| **Binary Search**    | O(1)          | O(log n)         | O(log n)       | O(1)                 |
| **Bubble Sort**      | O(n)          | O(n²)            | O(n²)          | O(1)                 |

---

## 📌 Explanations

### 1️⃣ **Linear Search**
- **Best Case:** O(1) if the target is the first element.
- **Average/Worst Case:** O(n) as it may need to check all `n` elements.
- **Space Complexity:** O(1) since it uses only a few variables.

### 2️⃣ **Binary Search**
- **Best Case:** O(1) if the target is the middle element on the first try.
- **Average/Worst Case:** O(log n) because it halves the search space at each step.
- **Space Complexity:** O(1) for the iterative version (recursive version would be O(log n) due to the call stack).

### 3️⃣ **Bubble Sort**
- **Best Case:** O(n) if the list is already sorted and optimized to stop early (not shown here).
- **Average/Worst Case:** O(n²) due to nested loops comparing and swapping elements.
- **Space Complexity:** O(1) as it sorts in place with no extra memory.

---

## 📌 Key Takeaways

1. **Linear Search:** 
   - Simple and works on unsorted lists.
   - Inefficient for large datasets due to O(n) time complexity.

2. **Binary Search:** 
   - Very fast (O(log n)) but requires the list to be sorted beforehand.
   - Ideal for large, sorted datasets.

3. **Bubble Sort:** 
   - Easy to understand and implement.
   - Inefficient (O(n²)) for large datasets, making it unsuitable for practical use.

---

## 📌 Conclusion

- Use **Linear Search** for small, unsorted datasets where simplicity is more important than speed.
- Use **Binary Search** for large, sorted datasets where efficiency is key.
- Use **Bubble Sort** primarily for learning purposes, as it is not efficient for real-world applications.