# 📌 3. Dive into the asyncio Library

"""
### Core Functions of `asyncio`:
1. **`asyncio.run()`**:
   - The entry point for running an async program.
   - Takes a coroutine (e.g., `main()`) and executes it.

2. **`asyncio.create_task()`**:
   - Schedules a coroutine to run concurrently as a task in the event loop.
   - Allows multiple coroutines to run in parallel.

3. **`asyncio.gather()`**:
   - Runs multiple coroutines concurrently and waits for all of them to complete.
   - Returns the results of all coroutines as a list.

### Exercise: Simulate Concurrent API Calls
Let’s simulate fetching data from multiple APIs concurrently using `asyncio.create_task()` and `asyncio.gather()`.
"""

import asyncio

# ----------------------------------------------
# Simulate Fetching Data from APIs
# ----------------------------------------------

async def fetch_data(name, delay):
    """
    Simulates fetching data from an API.
    Args:
        name (str): The name of the API.
        delay (int): The time (in seconds) to simulate the API response delay.
    Returns:
        str: A message indicating the data fetched from the API.
    """
    print(f"Fetching {name}...")
    await asyncio.sleep(delay)  # Simulate API delay
    print(f"Fetched {name}")
    return f"Data from {name}"

# ----------------------------------------------
# Main Function to Run Concurrent Tasks
# ----------------------------------------------

async def main():
    """
    Main coroutine to schedule and run multiple tasks concurrently.
    """
    # Schedule tasks concurrently using asyncio.create_task()
    task1 = asyncio.create_task(fetch_data("API 1", 2))
    task2 = asyncio.create_task(fetch_data("API 2", 3))
    
    # Wait for all tasks to finish and collect results using asyncio.gather()
    results = await asyncio.gather(task1, task2)
    print("Results:", results)

# Run the main coroutine
asyncio.run(main())

"""
### Output:
Fetching API 1...
Fetching API 2...
[2-second pause]
Fetched API 1
[1-second pause]
Fetched API 2
Results: ['Data from API 1', 'Data from API 2']

### Total Time:
~3 seconds.

### Explanation:
1. **`asyncio.create_task()`**:
   - Starts each coroutine immediately, allowing them to run in parallel.
   - Both `fetch_data("API 1")` and `fetch_data("API 2")` begin execution without waiting for each other.

2. **`asyncio.gather()`**:
   - Waits for all scheduled tasks to complete.
   - Collects the return values of all tasks and returns them as a list.

3. **Concurrency**:
   - The tasks overlap, so the total time is determined by the longest task (3 seconds), not the sum of all delays.

### Key Takeaways:
- `asyncio.create_task()` is used to schedule coroutines to run concurrently.
- `asyncio.gather()` is used to wait for all tasks to complete and collect their results.
- Asynchronous programming allows efficient handling of I/O-bound tasks, such as API calls, by overlapping wait times.
"""