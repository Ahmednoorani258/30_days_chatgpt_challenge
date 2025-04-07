# 4. Play with Hugging Face Transformers
# What Are Hugging Face Transformers?
# Hugging Face Transformers is a library built on deep learning models (like BERT, GPT, etc.) that excel at understanding and generating human language. These models are pre-trained on massive datasets (think billions of words) and then fine-tuned for specific tasks, making them slower but incredibly powerful compared to libraries like spaCy.

# Why Powerful?: They capture deep context—like how “bank” means “riverbank” or “money bank” based on surrounding words—something simpler models struggle with.
# Key Feature: The pipeline tool simplifies complex tasks into one-liners.
# Analogy: Think of Transformers as a genius librarian who’s read every book in the world and can instantly summarize, analyze feelings, or translate—while spaCy is a fast, practical assistant for basic tasks.

# Prerequisites
# Make sure you’ve installed the library:

# First intall Transformer
# pip install transformers

"""
Mini Task 1: Sentiment Analysis with Pipeline
What is Sentiment Analysis?
Sentiment analysis figures out the emotion or opinion in text—positive, negative, or neutral. It’s like guessing if a movie review is a thumbs-up or thumbs-down based on the words.

Use Case: Businesses use it to analyze customer feedback; chatbots use it to gauge your mood.
How the Pipeline Works
The pipeline in Hugging Face is a ready-to-use tool that loads a pre-trained model and tokenizer behind the scenes. For sentiment analysis, it defaults to a model like DistilBERT, fine-tuned to classify feelings.
"""

from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test it with a sentence
result = classifier("I love learning NLP with Hugging Face!")
print(result)

"""
Explanation
pipeline("sentiment-analysis"): Sets up a pre-trained model for sentiment. You don’t need to specify the model—it picks a good default (e.g., distilbert-base-uncased-finetuned-sst-2-english).
Input: The sentence "I love learning NLP with Hugging Face!".
Output:
label: POSITIVE—the sentiment is happy/positive.
score: 0.9998—a confidence score (0 to 1), super confident it’s positive!
How It Works Under the Hood
Tokenization: Breaks the sentence into tokens (e.g., "I", "love", "learning", etc.).
Model Processing: Feeds tokens into a Transformer (like BERT), which looks at context (e.g., "love" + "learning" = good vibes).
Prediction: Outputs a sentiment label and confidence score.
Another Example
"""

result = classifier("This task is so boring.")
print(result)
# Output: [{'label': 'NEGATIVE', 'score': 0.9987}]


# Why It’s Cool
# With one line, you get a deep-learning-powered sentiment detector—no need to train anything yourself!
















# Task2
# Mini Task 2: Text Summarization
# What is Text Summarization?
# Text summarization takes a long piece of text and condenses it into a shorter version, keeping the main ideas. It’s like summarizing a book into a paragraph—great for news, reports, or long articles.

# Types:
# Extractive: Picks key sentences as-is.
# Abstractive: Rewrites the text in its own words (what we’ll do here).
# Use Case: Auto-summarize emails or articles.
# How the Pipeline Works
# The summarization pipeline uses a pre-trained model (e.g., BART or T5) designed for abstractive summarization. It understands the text and generates a concise version.

from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Sample text
text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves tasks like text analysis, sentiment detection, and language generation. NLP powers tools like chatbots, search engines, and voice assistants, making them understand and respond to human input effectively.
"""

# Summarize it
result = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(result)

# Explanation
# pipeline("summarization"): Loads a default model (e.g., facebook/bart-large-cnn).
# Parameters:
# max_length=50: Limits the summary to ~50 tokens (words/punctuation).
# min_length=25: Ensures it’s at least ~25 tokens.
# do_sample=False: Uses deterministic output (no randomness).
# Input: A 3-sentence paragraph about NLP.
# Output: A single sentence capturing the essence—NLP’s focus and applications.
# How It Works Under the Hood
# Tokenization: Splits text into tokens.
# Encoding: Transformer model (e.g., BART) reads the full text, understanding context.
# Generation: Creates a new, shorter text by “rewriting” key points.
# Decoding: Turns the model’s output back into readable words.





# Detailed Explanation Detailed Breakdown: Why Transformers Shine
# Power of Deep Learning
# Context: Unlike spaCy’s rule-based or simpler models, Transformers (e.g., BERT) understand context across whole sentences. E.g., “I banked on it” (relied) vs. “I went to the bank” (place).
# Pre-Training: Models are trained on massive datasets (e.g., Wikipedia, books), then fine-tuned for tasks like sentiment or summarization.
# Trade-Off
# Speed: Slower than spaCy because of complex neural networks.
# Power: Way more accurate for tricky tasks requiring deep understanding.
# Example Comparison
# SpaCy Sentiment: No built-in sentiment tool—you’d need extra training.
# Transformers Sentiment: Pre-trained and ready with pipeline.