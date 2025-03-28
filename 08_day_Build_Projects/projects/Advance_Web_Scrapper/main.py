import streamlit as st
from scraper import CATEGORIES, Scrape_url

# Set page title and description
st.title('Advanced Web Scraper')
st.write('Enter URLs and select categories to scrape data from websites.')

# Sidebar for inputs
with st.sidebar:
    st.header('Input Parameters')
    urls = st.text_area(
        'Enter URLs (one per line):', 
        help='e.g., http://example.com'
    )
    categories = st.multiselect(
        'Select categories to scrape:', 
        options=list(CATEGORIES.keys()), 
        help='Choose the types of data to extract from the websites.'
    )
    scrape_button = st.button('Scrape')

# Main area for results and scraping logic
if scrape_button:
    if not urls:
        st.warning('Please enter some URLs.')
    elif not categories:
        st.warning('Please select some categories.')
    else:
        urls_list = [url.strip() for url in urls.split('\n') if url.strip()]
        total_urls = len(urls_list)
        
        # Initialize progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Dictionary to track total counts per category
        total_counts = {cat: 0 for cat in categories}
        
        # Scrape each URL with progress feedback
        for i, url in enumerate(urls_list):
            status_text.text(f'Scraping URL {i+1} of {total_urls}: {url}')
            try:
                data = Scrape_url(url, categories)
                # Use expander for each URL's results
                with st.expander(f'Results for URL: {url}'):
                    if data:
                        for category, items in data.items():
                            # Display category with item count
                            st.write(f"**{category.capitalize()} ({len(items)}):**")
                            if items:
                                # Make links clickable if category is 'links'
                                if category.lower() == 'links':
                                    for item in items:
                                        st.markdown(f"- [{item}]({item})")
                                else:
                                    for j, item in enumerate(items, 1):
                                        st.write(f"{j}. **- {item}**")
                            else:
                                st.write("No data found for this category.")
                            # Update total counts
                            if category in total_counts:
                                total_counts[category] += len(items)
                    else:
                        st.write("Failed to scrape or no data available.")
            except Exception as e:
                st.error(f"Error scraping {url}: {e}")
            # Update progress bar
            progress_bar.progress((i + 1) / total_urls)
        
        # Update status when done
        status_text.text('Scraping completed.')
        
        # Display summary
        st.markdown('---')
        st.header('Summary')
        st.write(f"Total URLs scraped: {total_urls}")
        for cat, count in total_counts.items():
            st.write(f"- {cat.capitalize()}: {count}")
        total_items = sum(total_counts.values())
        st.write(f"Total items found: {total_items}")