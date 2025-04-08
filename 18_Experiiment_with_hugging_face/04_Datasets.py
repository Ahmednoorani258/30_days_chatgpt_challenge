from datasets import load_dataset
# +_______________________________________________________
# +_______________________________________________________

# Load IMDB dataset (movie reviews)
dataset = load_dataset("imdb")

# Access the first training example
# print(dataset["train"][0])
# +_______________________________________________________
# +_______________________________________________________
# Explanation:
# Text: A movie review.
# Label: 0 (negative) or 1 (positive).
# Structure: The dataset has train, test, and unsupervised splits, each as a dictionary-like object.
# Preprocessing Example
# Tokenize the dataset for DistilBERT:

# Hugging Face Transformers library se AutoTokenizer import karte hain
from transformers import AutoTokenizer

# Step 1: DistilBERT tokenizer ko load karte hain
# Tokenizer ka kaam hai raw text ko token IDs mein convert karna jo model samajh sake.
# "distilbert-base-uncased" ek pre-trained tokenizer hai jo lowercased text ke liye optimized hai.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Step 2: Ek function define karte hain jo dataset ko tokenize karega
# Ye function ek example (text) lega aur usko tokenize karega.
# "padding='max_length'" ka matlab hai ke har input ko ek fixed length tak pad karein.
# "truncation=True" ka matlab hai ke agar text zyada lamba ho to usko truncate kar dein.
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Step 3: Dataset ko tokenize karte hain
# "dataset.map" ka use karte hain har example ko tokenize karne ke liye.
# "batched=True" ka matlab hai ke ek batch mein multiple examples ko process karein.
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Tokenized dataset ka output dekhte hain
# "train" dataset ke pehle example ke pehle 10 tokens print karte hain.
print(tokenized_dataset["train"][0]["input_ids"][:10])  # Pehle 10 tokens

# Model                Use Case                      Details
# --------------------------------------------------------------------------------
# distilbert           Fast classification           Smaller, distilled version of BERT.
# bert-base-uncased    General-purpose understanding  Bidirectional, 12 layers, 110M params.
# t5-small             Text-to-text (summarization, Q&A) Encoder-decoder, 60M params.
# gpt2                 Text generation               Decoder-only, autoregressive.
# whisper              Speech-to-text                Audio processing (not text-only).
# layoutLM             Document understanding        For PDFs/images with text layout.