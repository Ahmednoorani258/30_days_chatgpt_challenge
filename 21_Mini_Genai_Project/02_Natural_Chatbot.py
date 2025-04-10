
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch library ko import karte hain jo tensor operations aur GPU acceleration ke liye use hoti hai
import torch

# Step 1: DialoGPT ka tokenizer aur model load karte hain
# "microsoft/DialoGPT-small" ek pre-trained conversational model hai jo small size ka version hai
# Tokenizer ka kaam hai user input ko token IDs mein convert karna
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Model ka kaam hai tokenized input ko process karna aur response generate karna
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Step 2: Ek function define karte hain jo chatbot ka reply generate karega
def generate_reply(user_input):
    # User ke input ko token IDs mein convert karte hain
    # "eos_token" ka matlab hai end-of-sequence token jo model ko batata hai ke input khatam ho gaya
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")  # 'pt' PyTorch tensors ke liye hai

    # Model ko input dete hain aur response generate karte hain
    # "max_length=1000" ka matlab hai ke response ki maximum length 1000 tokens tak ho sakti hai
    # "pad_token_id" padding ke liye use hota hai agar input ki length chhoti ho
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Generated response ko wapas human-readable text mein decode karte hain
    # "skip_special_tokens=True" ka matlab hai ke special tokens (e.g., <eos>, <pad>) ko remove kar diya jaye
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Response ko return karte hain
    return response

# Step 3: Chatbot ko test karte hain
# User se input lete hain
you_say = input("What do you want to say? ")  # User se ek sentence ya question input karne ko kaha jata hai


# What’s happening?

# tokenizer: Turns your words into numbers (computers love numbers!).
# model.generate: Makes up a reply based on what you said.
# decode: Turns the numbers back into words.
# Example:

# You: “How’s your day?”
# Bot: “Pretty good, thanks! How’s yours?”
# Why it’s fun: It feels more like a real chat, but sometimes it says silly things!