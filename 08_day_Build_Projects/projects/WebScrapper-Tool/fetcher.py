import aiohttp


async def fetch_url(session, url):
    """Fetch HTML content from a URL asynchronously."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Failed to fetch {url}: Status {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


# Async Programming: async def defines an asynchronous function. await pauses execution until the I/O operation (HTTP request) completes.

# Context Manager: async with ensures the session is properly closed.

# Error Handling: Catches network errors or invalid responses.
