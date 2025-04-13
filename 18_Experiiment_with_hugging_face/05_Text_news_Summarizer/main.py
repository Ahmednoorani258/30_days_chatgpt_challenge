import streamlit as st
from transformers import pipeline


# Function to load models and cache them for efficiency
@st.cache_resource
def load_pipeline(task, model_name):
    return pipeline(task, model=model_name)


# App title
st.title("Sentiment Analysis with Hugging Face Transformer")
st.write(
    "This app performs sentiment analysis using pre-trained model from Hugging Face Transformer."
)

# Task selection dropdown
task = st.selectbox(
    "Choose an NLP Task",
    [
        "Sentiment Analysis",
    ],
)

# Model selection based on task
if task == "Sentiment Analysis":
    model_name = st.selectbox(
        "Choose a Model",
        [
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
    )

# Button to load the model
if st.button("Load Model"):
    with st.spinner("Loading model..."):
        try:
            classifier = load_pipeline(task.lower().replace(" ", "-"), model_name)
            st.session_state.classifier = classifier
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Input and prediction section
if "classifier" in st.session_state:
    if task == "Sentiment Analysis":
        text = st.text_input(
            "Enter text for sentiment analysis:", "I love using Hugging Face!"
        )
        if st.button("Analyze Sentiment"):
            result = st.session_state.classifier(text)
            st.write(
                f"Sentiment: **{result[0]['label']}** (Confidence: {result[0]['score']:.4f})"
            )
else:
    st.write("Please load a model to proceed.")

# Prediction history
if "predictions" not in st.session_state:
    st.session_state.predictions = []

if st.button("Save Prediction") and "result" in locals():
    if task == "Sentiment Analysis":
        st.session_state.predictions.append(
            {"Task": task, "Text": text, "Sentiment": result[0]["label"]}
        )

    st.write("Prediction saved to history.")

# Display prediction history
if st.session_state.predictions:
    st.subheader("Prediction History")
    for pred in st.session_state.predictions:
        st.write(pred)
