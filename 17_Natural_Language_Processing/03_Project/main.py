import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline with distilbert
@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

classifier = load_classifier()

# Streamlit App
st.title("Twitter Sentiment Classifier")
st.write("Enter a tweet to classify its sentiment (positive or negative).")

# Initialize session state for history
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

# User input
tweet = st.text_input("Enter a tweet:", "I love this sunny day!")

# Analyze sentiment
if st.button("Analyze Sentiment"):
    if tweet:
        with st.spinner("Analyzing..."):
            result = classifier(tweet)
            label = result[0]["label"]
            score = result[0]["score"]
            
            # Store in history
            st.session_state.sentiment_history.append({
                "tweet": tweet,
                "sentiment": label,
                "confidence": score
            })
            
            # Display result
            st.subheader("Result")
            st.write(f"Sentiment: **{label}** (Confidence: {score:.4f})")
    else:
        st.warning("Please enter a tweet!")

# Display history
if st.session_state.sentiment_history:
    st.subheader("Sentiment History")
    for entry in reversed(st.session_state.sentiment_history):
        st.write(f"- Tweet: '{entry['tweet']}' | Sentiment: **{entry['sentiment']}** | Confidence: {entry['confidence']:.4f}")

# Clear history button
if st.button("Clear History"):
    st.session_state.sentiment_history = []
    st.write("History cleared!")