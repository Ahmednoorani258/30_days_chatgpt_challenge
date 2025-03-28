import requests
from bs4 import BeautifulSoup

def fetch_html(url):
    try:
        response = requests.get(url,timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {url}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def Extract_titles(soup):
    return [title.get_text() for title in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) or soup.find_all('title')]

def Extract_images(soup):
    return [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]

def Extract_links(soup):
    return [a['href'] for a in soup.find_all('a') if 'href' in a.attrs]

def Extract_paragraphs(soup):
    return [p.get_text(strip=True) for p in soup.find_all('p')]

CATEGORIES = {
    'titles': Extract_titles,
    'images': Extract_images,
    'links': Extract_links,
    'paragraphs': Extract_paragraphs
}


def Scrape_url(url,categories):
    html = fetch_html(url)
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        result = {}
        for category in categories:
            result[category] = CATEGORIES[category](soup)
        return result
    else:
        return None