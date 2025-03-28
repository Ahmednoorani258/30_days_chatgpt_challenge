# Project 1: Async Web Scraper

## 📌 Scope and Requirements

### Description:
A tool that scrapes article titles from multiple websites concurrently.

### Inputs:
- A list of URLs (we’ll use mock or scraper-friendly sites to avoid legal issues).

### Outputs:
- A list of titles printed to the console or saved to a file.

### Key Functionalities:
1. Fetch HTML content asynchronously from multiple URLs.
2. Parse HTML to extract titles.
3. Handle errors (e.g., failed requests).

---

## 🛠 Step 1: Set Up the Environment

### Create a Directory:
Run the following commands to create and navigate to your project directory:
```bash
mkdir async_web_scraper
cd async_web_scraper

### Initialize Git

git init
echo "# Async Web Scraper" > README.md
git add README.md
git commit -m "Initial commit"

# Set Up Virtual Environment:

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install aiohttp beautifulsoup4

# aiohttp: For async HTTP requests.
# beautifulsoup4: For HTML parsing.
