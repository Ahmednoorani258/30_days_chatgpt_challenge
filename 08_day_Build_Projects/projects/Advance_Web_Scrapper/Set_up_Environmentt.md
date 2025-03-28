# Project 1: Advanced Web Scraper

## 📌 Scope and Requirements

### Description:
A web application that scrapes multiple URLs and extracts data based on user-selected categories (e.g., titles, images, links, paragraphs).

### Inputs:
- URLs (one per line).
- Categories to scrape (e.g., titles, images, links, paragraphs).

### Outputs:
- Scraped data displayed in the Streamlit app.
- Data organized by URL and category.

### Advanced Features:
- Parses full HTML category-wise.
- Allows users to select specific data to extract.

---

## 🛠 Step 1: Set Up the Environment

### Create a Directory:
Run the following commands to create and navigate to your project directory:
```bash
mkdir web_scraper
cd web_scraper
Initialize UV Project:

uv init
Add Dependencies:

uv add requests beautifulsoup4 streamlit
requests: For fetching HTML content.
beautifulsoup4: For parsing HTML.
streamlit: For the web interface.