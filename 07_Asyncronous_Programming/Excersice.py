# 📌 6. Bonus Challenge (Optional): Refactor a Synchronous Scraper

"""
### Task:
Convert the following synchronous scraper into an asynchronous version using `aiohttp` and `asyncio`.

---

### Synchronous Scraper:
The original scraper fetches data from multiple URLs sequentially using the `requests` library.

```python
import requests
import time
"""


start = time.time()
urls = ["https://example.com", "https://example.org"]
for url in urls:
    response = requests.get(url)
    print(f"Fetched {len(response.text)} characters from {url}")
print("Time:", time.time() - start)


"""
# 📌 6. Bonus Challenge (Optional): Refactor a Synchronous Scraper


### Objective:
Refactor the synchronous scraper into an asynchronous version using `aiohttp` and `asyncio` to improve performance when fetching data from multiple URLs.

The asynchronous version will:
1. Fetch data from multiple URLs concurrently.
2. Use `aiohttp.ClientSession()` for efficient HTTP connection management.
3. Leverage `asyncio.gather()` to run multiple tasks concurrently.

---

### Implementation:
Below is the refactored asynchronous scraper.
"""

import aiohttp
import asyncio
import time

# ----------------------------------------------
# Async Function to Fetch Data from a URL
# ----------------------------------------------


async def fetch_url(url):
    """
    Fetches the content of the given URL asynchronously.
    Args:
        url (str): The URL to fetch data from.
    Returns:
        str: The text content of the response.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


# ----------------------------------------------
# Main Function to Fetch Data Concurrently
# ----------------------------------------------


async def main():
    """
    Main coroutine to fetch data from multiple URLs concurrently.
    """
    urls = ["https://example.com", "https://example.org"]

    # Measure the start time
    start = time.time()

    # Create tasks for each URL and run them concurrently
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    # Process and display the results
    for url, result in zip(urls, results):
        print(f"Fetched {len(result)} characters from {url}")

    # Measure and display the total time taken
    print("Time:", time.time() - start)


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())

"""
### Output (Asynchronous Version):
# Fetched 1256 characters from https://example.com
# Fetched 1350 characters from https://example.org
# Time: ~1.5 seconds (varies based on network speed)

# ---

### Explanation:
# 1. **Concurrency**:
#    - The asynchronous version fetches data from multiple URLs concurrently, reducing the total execution time compared to the sequential version.

# 2. **`aiohttp.ClientSession()`**:
#    - Manages HTTP connections efficiently and ensures proper cleanup using `async with`.

# 3. **`asyncio.gather()`**:
#    - Runs multiple coroutines concurrently and waits for all of them to complete.

# ---

### Comparison with Synchronous Version:
# 1. **Synchronous Version**:
#    - Fetches URLs one by one, resulting in a total time equal to the sum of all request durations.

# 2. **Asynchronous Version**:
#    - Fetches URLs concurrently, resulting in a total time equal to the longest request duration.

# ---

### Key Takeaways:
# - Asynchronous programming significantly improves performance for I/O-bound tasks like web scraping.
# - Using `aiohttp` and `asyncio` allows efficient handling of multiple requests concurrently.
# """
