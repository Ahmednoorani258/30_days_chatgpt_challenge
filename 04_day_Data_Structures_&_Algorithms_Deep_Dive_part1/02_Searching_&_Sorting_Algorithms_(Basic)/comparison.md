# 🔍 Comparison of Linear Search, Binary Search, and Bubble Sort

This document provides a comparison of three fundamental algorithms: **Linear Search**, **Binary Search**, and **Bubble Sort**. Each algorithm serves a specific purpose and has its own strengths and weaknesses.

---

## 📌 Algorithm Comparison Table

| **Aspect**           | **Linear Search**               | **Binary Search**               | **Bubble Sort**                 |
|-----------------------|----------------------------------|----------------------------------|----------------------------------|
| **Purpose**           | Find an element                | Find an element                 | Sort a list                     |
| **Input Requirement** | Any list                       | Sorted list                     | Any list                        |
| **Time Complexity**   | O(n) (worst)                   | O(log n) (worst)                | O(n²) (worst)                   |
| **Space Complexity**  | O(1)                           | O(1)                            | O(1)                            |
| **Efficiency**        | Slow for large lists           | Fast for large sorted lists     | Slow for large lists            |
| **Best For**          | Small/unsorted lists           | Large sorted lists              | Learning/small lists            |

---

## 📌 Summary of Algorithms

### 1️⃣ **Linear Search**
- **Description:** A basic way to find an element in any list by checking each element one by one.
- **Strengths:** Easy to implement, works on unsorted data.
- **Weaknesses:** Slow for large datasets due to O(n) time complexity.

### 2️⃣ **Binary Search**
- **Description:** A faster search method for sorted lists, cutting the search space in half at each step.
- **Strengths:** Very efficient for large sorted lists with O(log n) time complexity.
- **Weaknesses:** Requires the list to be sorted beforehand.

### 3️⃣ **Bubble Sort**
- **Description:** A simple sorting algorithm that repeatedly swaps adjacent elements if they are in the wrong order.
- **Strengths:** Easy to understand and implement, good for teaching purposes.
- **Weaknesses:** Inefficient for large datasets due to O(n²) time complexity.

---

## 📌 Conclusion
- **Linear Search** is best for small or unsorted lists where simplicity is more important than speed.
- **Binary Search** is ideal for large, sorted lists where efficiency is key.
- **Bubble Sort** is primarily used for learning purposes and is not suitable for practical use on large datasets.