# ✅ 2. Use Pre-Trained Transformers via Pipeline
# Hugging Face’s pipeline API is a high-level tool for quick NLP tasks. It bundles a model, tokenizer, and preprocessing into one easy function.

# How to Use Pipelines
# Import the pipeline function.
# Specify the task (e.g., "sentiment-analysis").
# Pass your text and get results.
from transformers import pipeline

model = "distilbert-base-uncased-finetuned-sst-2-english"
# Load the pipeline with a specific model
sentiment = pipeline("sentiment-analysis", model=model)

# Analyze a sentence
result = sentiment("I love using modern AI tools!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Explanation: The model predicts the sentiment as positive with 99.98% confidence.

ner = pipeline("ner", grouped_entities=True, model=model)
result = ner("Hugging Face Inc. is based in New York City.")
print(result)

summarizer = pipeline("summarization")
text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves tasks like text analysis, sentiment detection, and language generation. NLP powers tools like chatbots, search engines, and voice assistants, making them understand and respond to human input effectively.
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary)



qa = pipeline("question-answering")
question = "Where is Hugging Face based?"
context = "Hugging Face Inc. is based in New York City."
result = qa(question=question, context=context)
print(result)