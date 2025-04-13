# 📌 5. Build a Mini Project: Asynchronous Web Scraper

"""
### Project Outline:
**Objective**: Fetch content from multiple URLs concurrently.

### Steps:
1. Define a list of URLs to scrape.
2. Create an async function to fetch data from each URL.
3. Use `asyncio.gather()` to run all fetches concurrently.
4. Process and display the results.

This project demonstrates the power of asynchronous programming by fetching data from multiple URLs in parallel, significantly reducing the total execution time compared to sequential requests.
"""

import aiohttp
import asyncio

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
# Main Function to Fetch and Process Data
# ----------------------------------------------


async def main():
    """
    Main coroutine to fetch data from multiple URLs concurrently and process the results.
    """
    # List of URLs to scrape
    urls = ["https://example.com", "https://example.org", "https://example.net"]

    # Create a list of tasks for each URL
    tasks = [fetch_url(url) for url in urls]

    # Run all tasks concurrently and wait for their completion
    results = await asyncio.gather(*tasks)

    # Process and display the results
    for url, result in zip(urls, results):
        print(f"Fetched {len(result)} characters from {url}")


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())

"""
### Sample Output:
Fetched 1256 characters from https://example.com
Fetched 1350 characters from https://example.org
Fetched 1420 characters from https://example.net

---

### Explanation:
1. **List of URLs**:
   - A list of URLs is defined to simulate scraping multiple websites.

2. **Task Creation**:
   - A list comprehension `[fetch_url(url) for url in urls]` creates a coroutine for each URL.

3. **Concurrent Execution**:
   - `asyncio.gather(*tasks)` runs all coroutines concurrently, waiting only as long as the slowest response.

4. **Processing Results**:
   - The `zip(urls, results)` pairs each URL with its corresponding result, allowing us to process and display the fetched data.

---

### Key Concepts Demonstrated:
1. **Concurrency with `asyncio.gather()`**:
   - Runs multiple coroutines concurrently, reducing the total execution time.

2. **Efficient Resource Management with `aiohttp.ClientSession()`**:
   - Manages HTTP connections efficiently and ensures proper cleanup using `async with`.

3. **Scalability**:
   - This approach scales well for a large number of URLs, as tasks are executed concurrently.

---

### Possible Enhancements:
1. **Error Handling**:
   - Add error handling to manage failed requests or timeouts.
   - Example:
     ```python
     async with session.get(url) as response:
         if response.status == 200:
             return await response.text()
         else:
             return f"Error: {response.status}"
     ```

2. **Data Extraction**:
   - Instead of measuring content length, extract specific data (e.g., titles, meta descriptions) using libraries like `BeautifulSoup`.

3. **Save Results**:
   - Save the fetched data to a file or database for further analysis.

4. **Rate Limiting**:
   - Implement rate limiting to avoid overwhelming servers with too many requests.

---

### Benefits of Asynchronous Web Scraping:
1. **Efficiency**:
   - Fetching multiple URLs concurrently reduces the total execution time compared to sequential requests.

2. **Scalability**:
   - Handles a large number of requests efficiently without blocking the event loop.

3. **Flexibility**:
   - Easily extendable to include advanced features like error handling, retries, and data extraction.
"""
