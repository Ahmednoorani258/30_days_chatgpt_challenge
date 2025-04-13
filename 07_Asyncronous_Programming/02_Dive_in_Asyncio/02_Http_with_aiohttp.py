# 📌 4. Implement Async HTTP Requests with aiohttp

"""
### Objective:
Learn how to make asynchronous HTTP requests using the `aiohttp` library. This allows you to perform non-blocking API calls,
making it ideal for fetching data from multiple URLs concurrently.

### Setup:
Install `aiohttp` if you haven’t already:
```bash
pip install aiohttp
"""

"""
### Key Concepts:
1. **`aiohttp.ClientSession()`**:
   - Manages connections efficiently for asynchronous HTTP requests.
   - Ensures proper resource cleanup when used with `async with`.

2. **`async with`**:
   - Ensures that resources (like HTTP sessions) are properly closed after use.

3. **`await response.json()`**:
   - Retrieves the response body as JSON without blocking other tasks.

---

### Practical Exercise: Async API Fetcher
Fetch data from an API endpoint asynchronously and process the response.
"""

import aiohttp
import asyncio

# ----------------------------------------------
# Async Function to Fetch Data from a URL
# ----------------------------------------------


async def fetch_url(url):
    """
    Fetches data from the given URL asynchronously.
    Args:
        url (str): The URL to fetch data from.
    Returns:
        dict: The JSON response from the API.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()


# ----------------------------------------------
# Main Function to Fetch and Process Data
# ----------------------------------------------


async def main():
    """
    Main coroutine to fetch data from an API and process the response.
    """
    url = "https://jsonplaceholder.typicode.com/posts/1"
    data = await fetch_url(url)
    print("Title:", data["title"])


# Run the main coroutine
asyncio.run(main())

"""
### Output:
Title: sunt aut facere repellat provident occaecati excepturi optio reprehenderit

---

### Explanation:
1. **`aiohttp.ClientSession()`**:
   - Manages HTTP connections efficiently for asynchronous requests.
   - Ensures proper resource cleanup when used with `async with`.

2. **`async with session.get(url)`**:
   - Sends a GET request to the specified URL.
   - Ensures the connection is properly closed after the request is complete.

3. **`await response.json()`**:
   - Retrieves the response body as JSON without blocking other tasks.

---

### Comparison with Synchronous Requests:
Let’s compare the performance of `aiohttp` with the synchronous `requests` library.
"""

# ----------------------------------------------
# Synchronous Request Example
# ----------------------------------------------

import requests
import time


def fetch_url_sync(url):
    """
    Fetches data from the given URL synchronously.
    Args:
        url (str): The URL to fetch data from.
    Returns:
        dict: The JSON response from the API.
    """
    response = requests.get(url)
    return response.json()


# Measure execution time for a synchronous request
start = time.time()
url = "https://jsonplaceholder.typicode.com/posts/1"
data = fetch_url_sync(url)
print("Title (Sync):", data["title"])
print("Time (Sync):", time.time() - start)

"""
### Output (Synchronous):
Title (Sync): sunt aut facere repellat provident occaecati excepturi optio reprehenderit
Time (Sync): ~0.5 seconds (varies based on network speed)

---

### Explanation:
- For a single request, the `requests` library works fine.
- However, fetching multiple URLs sequentially adds up the wait times, making it inefficient.

---

### Key Takeaways:
1. **`aiohttp`**:
   - Ideal for making non-blocking HTTP requests.
   - Efficiently manages connections using `async with aiohttp.ClientSession()`.

2. **Concurrency**:
   - Use `asyncio` to fetch data from multiple URLs concurrently.
   - Greatly improves performance for I/O-bound tasks like API calls.

3. **Comparison**:
   - Synchronous requests (`requests`) are fine for single requests but become inefficient for multiple sequential requests.
   - Asynchronous requests (`aiohttp`) excel at handling multiple requests concurrently, reducing overall execution time.
"""
