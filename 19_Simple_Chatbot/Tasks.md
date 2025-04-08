# 🎯 Day 19: Build a Simple Chatbot

## Goal of Day 19
Combine everything you’ve learned in NLP and Hugging Face to build a working chatbot — one that understands user intent and gives meaningful responses using pre-trained transformer models.

---

## 🧠 What You Will Learn Today
- How chatbots work: **intent recognition** and **response generation**.
- Using **Hugging Face transformers** in a chatbot workflow.
- Text classification + retrieval or rule-based response generation.
- **Bonus**: Build a simple **Streamlit UI** (optional but 🔥 for your portfolio).

---

## 🧰 Tools & Libraries

| **Tool**       | **Purpose**                                   |
|-----------------|-----------------------------------------------|
| **transformers** | NLP backbone (e.g., intent detection)        |
| **datasets**    | Optional dataset loading                     |
| **torch**       | Model backend                                |
| **streamlit**   | Web UI for chatbot                           |
| **langchain** (optional) | For chaining logic (2025 hot tool)  |

### Installation:
```bash
pip install transformers torch streamlit
```

🧩 Step-by-Step Tasks
✅ 1. Understand Chatbot Architecture
Basic rule-based NLP chatbot has 3 parts:

Input → Intent Detection (using transformer classifier or keyword matching)

Intent → Response Generation (rule-based or model-generated)

Reply → Sent back to user

Example:


User: What's the weather?
Bot: Let me check the forecast for you.
✅ 2. Intent Classification with Transformers
Use a pre-trained model like:

<!-- from transformers import pipeline
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
classifier("I'm feeling really happy today!") -->
🎯 Output:

json
Copy
Edit
[{'label': 'joy', 'score': 0.96}]
You can classify intents like: "greeting", "help", "weather", "goodbye" etc.

✅ 3. Design a Simple Intent-to-Response System
python
Copy
Edit
def get_response(label):
    responses = {
        "joy": "That's wonderful! 😄",
        "sadness": "I'm here for you. 💙",
        "greeting": "Hi there! How can I help?",
        "goodbye": "Take care! 👋",
        "anger": "Let's take a deep breath together. 🧘"
    }
    return responses.get(label.lower(), "I'm not sure how to respond to that.")
✅ 4. Optional: Add a Text Generation Model
Use a model like "microsoft/DialoGPT-small" for generating responses from scratch.

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

user_input = "How are you today?"
input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(response)
✅ 5. Bonus: Build a Streamlit Web Chat UI (🔥 Portfolio Booster)
python
Copy
Edit
# Save this as chatbot.py
import streamlit as st

st.title("Simple AI Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    label = classifier(user_input)[0]["label"]
    response = get_response(label)
    st.text_area("Bot:", response)
Run it:

bash
Copy
Edit
streamlit run chatbot.py
🧪 Mini Project Ideas
Emotion-Aware Chatbot
→ Detects emotions and responds empathetically
Use model: bhadresh-savani/bert-base-uncased-emotion

Mental Health Check-In Bot
→ Classifies user mood and gives helpful resources

Job Interview Bot
→ Uses classification to answer or redirect job-related questions

✅ Checklist for Day 19
Task	Done
Understood chatbot flow and components	✅
Used a transformer model to classify intent	✅
Mapped intents to responses	✅
(Optional) Used DialoGPT for response generation	✅
(Optional) Created a Streamlit UI for the chatbot	✅
Completed your first real AI Assistant 🎉	✅
