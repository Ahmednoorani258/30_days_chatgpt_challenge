import streamlit as st
from transformers import pipeline
import torch

classifier = pipeline("text-classification", model = "bhadresh-savani/bert-base-uncased-emotion" )

def get_response (emotion):
    responses = {
    "joy": "That’s awesome! 😄",
    "sadness": "I’m here for you. 💙",
    "greeting": "Hi there! How can I help?",
    "goodbye": "See ya later! 👋",
    "anger": "Let’s chill out together. 🧘",
    "fear": "It’s okay, take a deep breath. You’re not alone. 🌟",
    "surprise": "Wow, that’s unexpected! 🤯",
    "help": "Sure, let me know how I can assist you. 🛠️",
    "confusion": "I’ll do my best to clarify things for you. 🤔",
    "gratitude": "You’re welcome! I’m happy to help. 😊",
    "curiosity": "That’s an interesting question! Let me think... 🤓",
    "boredom": "Let’s find something fun to do! 🎮",
    "love": "Aww, that’s so sweet! ❤️",
    "neutral": "Got it. Let me know if there’s anything else. 🙂",
    "weather": "I can’t check the weather right now, but it’s always a good day to learn something new! 🌤️",
    "joke": "Why don’t scientists trust atoms? Because they make up everything! 😂",
    "unknown": "I’m not sure how to respond to that, but I’m here to help! 🤖"
    }
    return responses.get(emotion.lower(), "Hmm, I don’t know what to say!")
# Streamlit app


st.title("Simple Chatbot")
user_input = st.text_input("You: ", "Hello! How are you?")

if user_input:
    result = classifier(user_input)[0]
    emotion = result["label"]
    reply = get_response(emotion)
    st.text_area("Bot: ", reply, height=100)