import asyncio
import aiohttp
from fetcher import fetch_url
from parser import parse_titles

URLs = [
    'https://portfolio-tailwindcss-dun.vercel.app/'
]
async def scrape_url(session, url):
    html = await fetch_url(session, url)
    titles = parse_titles(html)
    return url, titles

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_url(session, url) for url in URLs]
        results = await asyncio.gather(*tasks)
        for url, titles in results:
            print(f"\nTitles from {url}:")
            for title in titles:
                print(f" - {title}")

if __name__ == "__main__":
    asyncio.run(main())