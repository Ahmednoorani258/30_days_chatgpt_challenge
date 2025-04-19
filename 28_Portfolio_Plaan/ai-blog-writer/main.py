from dotenv import load_dotenv
import os
import streamlit as st
import requests

# Load environment variables
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Set up Streamlit page config
st.set_page_config(page_title="AI Blog Writer", page_icon=":guardsman:", layout="wide")

# Sidebar with instructions
with st.sidebar:
    st.markdown("### Instructions")
    st.write("1. Enter a topic\n2. Choose tone\n3. Click Generate")

# Title of the app
st.title("AI Blog Writer")

# Input fields for topic and tone
topic = st.text_input("Enter your blog title", key="title")
tone = st.selectbox("Select tone:", ["Professional", "Casual", "Storytelling", "Friendly", "Inspirational"])

# Prepare the API request headers and data
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Default values for request data
data = {
    "model": "command",
    "prompt": f"Write a {tone.lower()} blog post about: {topic}",
    "max_tokens": 500,
    "temperature": 0.7,
}

# Button to trigger blog generation
if st.button("Generate Blog"):
    if topic:  # Make sure topic is not empty
        try:
            # API request to Cohere
            response = requests.post("https://api.cohere.ai/generate", headers=headers, json=data)
            
            # Check if the response was successful
            if response.status_code == 200:
                # Parse the response
                result = response.json()
                
                # Check if 'text' is in the response
                if "text" in result:
                    generated_blog = result["text"]
                    
                    # Display the generated blog text
                    st.text_area("Generated Blog", value=generated_blog, height=300)
                    
                    # Add download button
                    st.download_button("Download Blog", data=generated_blog, file_name="blog.txt")
                else:
                    st.error("No blog text generated. Please try again later.")
            else:
                st.error("Error generating blog. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a blog title before generating.")
