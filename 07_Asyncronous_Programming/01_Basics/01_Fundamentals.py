# 📌 1. Understand the Basics of Asynchronous Programming

"""
### Concept Overview: Synchronous vs. Asynchronous

#### Synchronous Execution (Blocking):
- In synchronous programming, code runs sequentially, one line at a time.
- Each task must complete before the next one begins.
- This is fine for simple scripts, but it becomes inefficient when tasks involve waiting—like downloading a file or querying an API.
- The program sits idle during these delays.

#### Asynchronous Execution (Non-Blocking):
- Asynchronous programming allows multiple tasks to run concurrently.
- While one task waits (e.g., for a server response), others can proceed.
- This is perfect for I/O-bound tasks (e.g., network requests, file operations), where the program spends significant time waiting for external resources.

### Exercise: Simulate Blocking vs. Non-Blocking Tasks
Let’s write a script to see the difference in action.
"""

# ----------------------------------------------
# Synchronous Example
# ----------------------------------------------

import time


def sync_task(name, duration):
    """Simulates a blocking task."""
    print(f"Starting {name}")
    time.sleep(duration)  # Simulates a blocking task (e.g., waiting for a server)
    print(f"Finished {name}")


# Run tasks sequentially
print("Synchronous Execution:")
sync_task("Task 1", 2)  # Takes 2 seconds
sync_task("Task 2", 3)  # Takes 3 seconds

# Total time: ~5 seconds.
# "Task 2" waits until "Task 1" finishes, even though the program is just idling during time.sleep().

# ----------------------------------------------
# Asynchronous Refactor
# ----------------------------------------------

import asyncio


async def async_task(name, duration):
    """Simulates a non-blocking task."""
    print(f"Starting {name}")
    await asyncio.sleep(duration)  # Non-blocking sleep
    print(f"Finished {name}")


async def main():
    """Runs multiple async tasks concurrently."""
    await asyncio.gather(async_task("Task 1", 2), async_task("Task 2", 3))


print("\nAsynchronous Execution:")
asyncio.run(main())

# Total time: ~3 seconds.
# Both tasks start immediately, and while one "sleeps," the other progresses.
# The total time is determined by the longest task (3 seconds), not the sum.

# ----------------------------------------------
# Explanation
# ----------------------------------------------

"""
1. **Synchronous Execution**:
   - Uses `time.sleep()` to simulate a blocking task.
   - Tasks run sequentially, and the total time is the sum of all task durations.

2. **Asynchronous Execution**:
   - Uses `asyncio.sleep()` to simulate a non-blocking task.
   - Tasks run concurrently, and the total time is determined by the longest task.

3. **Key Functions**:
   - `asyncio.sleep()`: The async equivalent of `time.sleep()`, allowing the event loop to switch to other tasks during the wait.
   - `asyncio.gather()`: Runs multiple coroutines concurrently, collecting their results when all are done.

### Key Takeaway:
Asynchronous programming is ideal for I/O-bound tasks where the program spends significant time waiting for external resources.
"""
