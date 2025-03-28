# Async Web Scraper

A tool that scrapes titles from multiple websites concurrently using Python's asyncio and aiohttp.

## Installation
1. Clone the repo: `git clone <repo-url>`
2. Set up virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install aiohttp beautifulsoup4`

## Usage
Run: `python main.py`

## Features
- Concurrent HTTP requests with aiohttp.
- HTML parsing with BeautifulSoup.
- Error handling for failed requests.