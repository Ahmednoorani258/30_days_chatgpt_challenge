# 📌 2. Learn and Practice Async/Await Syntax

"""
### Syntax Essentials:
1. **`async def`**:
   - Defines an asynchronous function (coroutine) that can be paused and resumed.
2. **`await`**:
   - Pauses the coroutine until the awaited task completes, freeing the event loop to handle other tasks.

### Exercise: Create and Chain Async Functions
Let’s explore how to define and use async functions with `await`, and how to run them sequentially or concurrently.
"""

import asyncio

# ----------------------------------------------
# Simple Async Function
# ----------------------------------------------

async def wait_and_print(message, delay):
    """Wait for a given delay and then print a message."""
    await asyncio.sleep(delay)
    print(message)

# Sequential Execution
async def main_sequential():
    """Run async functions sequentially."""
    await wait_and_print("Hello after 2 seconds", 2)
    await wait_and_print("Hello after another 3 seconds", 3)

print("Sequential Execution:")
asyncio.run(main_sequential())

"""
### Output (Sequential Execution):
[2-second pause]
Hello after 2 seconds
[3-second pause]
Hello after another 3 seconds

Explanation:
- Each `await` pauses the `main_sequential()` coroutine until the previous task completes.
- The tasks run one after the other, resulting in a total time of ~5 seconds.
"""

# ----------------------------------------------
# Concurrent Execution
# ----------------------------------------------

async def main_concurrent():
    """Run async functions concurrently using asyncio.gather()."""
    await asyncio.gather(
        wait_and_print("Hello after 2 seconds", 2),
        wait_and_print("Hello after 3 seconds", 3)
    )

print("\nConcurrent Execution:")
asyncio.run(main_concurrent())

"""
### Output (Concurrent Execution):
[2-second pause]
Hello after 2 seconds
[1-second pause]
Hello after 3 seconds

Explanation:
- `await` suspends the coroutine, but when used with `asyncio.gather()`, it schedules all tasks to run concurrently in the event loop.
- The tasks overlap, so the total time is determined by the longest task (~3 seconds), not the sum of all delays.
"""

# ----------------------------------------------
# Key Takeaways
# ----------------------------------------------

"""
1. **Sequential Execution**:
   - Tasks run one after the other.
   - Total time = Sum of all delays.

2. **Concurrent Execution**:
   - Tasks run concurrently using `asyncio.gather()`.
   - Total time = Longest delay among all tasks.

3. **Power of Async**:
   - By overlapping wait times, asynchronous programming allows efficient use of resources, especially for I/O-bound tasks.
"""