from bs4 import BeautifulSoup

def parse_titles(html):
    """Parse HTML and extract article titles."""
    if html is None:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    titles = [title.get_text() for title in soup.find_all('h1') or soup.find_all('title')]
    
    return titles if titles else []

# Data Structures: Returns a list of titles. Uses list comprehension for efficiency.
# Note: Parsing is synchronous because it’s CPU-bound and fast, unlike fetching.