# 🚀 Day 18: Experiment with Hugging Face Transformers

## 🎯 Goal of Day 18
Learn how to use **Hugging Face Transformers** for real-world NLP tasks like text classification, summarization, and question answering, using both pipelines and custom models with a PyTorch backend.

---

## 🧠 What You Will Learn
- What transformer models are and why they matter.
- Key concepts: **Tokenizers**, **Model Architecture**, **Attention**.
- How to use Hugging Face transformers with real data.
- Load and fine-tune pre-trained models for text tasks.
- Use the **datasets** library to work with high-quality NLP datasets.

---

## 🔧 Libraries You’ll Use Today

| **Purpose**           | **Tool**                   |
|------------------------|----------------------------|
| **Model Hub & APIs**   | transformers               |
| **Datasets**           | datasets                  |
| **Training Backend**   | torch (PyTorch)           |
| **Environment**        | Jupyter Notebook/Colab    |

### Installation:
```bash
pip install transformers datasets torch
```

🧩 Step-by-Step Tasks
✅ 1. Understand Transformers
Transformers are deep learning models that:

Process entire sentences at once (not word-by-word).
Use self-attention to understand word relationships.
Power models like BERT, RoBERTa, GPT, T5, DistilBERT, etc.
📘 Suggested Reading:

Transformers Explained – Simple
BERT vs GPT vs T5 vs RoBERTa Comparison
✅ 2. Use Pre-Trained Transformers via Pipeline
Try these 4 pipelines and observe the outputs:

```python

from transformers import pipeline

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love using modern AI tools!"))

# Named Entity Recognition (NER)
ner = pipeline("ner", grouped_entities=True)
print(ner("Hugging Face Inc. is based in New York City."))

# Summarization
summarizer = pipeline("summarization")
print(summarizer("Long article text here..."))

# Question Answering
qa = pipeline("question-answering")
print(qa(question="Where is Hugging Face based?", context="Hugging Face Inc. is based in New York City."))

```

✅ 3. Load and Use a Specific Model
You can directly choose models from the 🤗 Model Hub.

Example: Sentiment classifier using DistilBERT:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("AI is changing the world!", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted = torch.argmax(logits)
print(predicted)
```


✅ 4. Work with Real Datasets using datasets
The datasets library makes it easy to load and work with large datasets.

Example: Load the IMDB movie reviews dataset:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")  # Movie reviews dataset
print(dataset["train"][0])
```

# ✅ 5. Explore Model Types

Choose a model based on your use case:

| **Model**             | **Use Case**                              |
|------------------------|-------------------------------------------|
| **distilbert**         | Fast + small for classification           |
| **bert-base-uncased**  | General-purpose understanding             |
| **t5-small**           | Text-to-text: summarization, Q&A          |
| **gpt2**               | Text generation                          |
| **whisper**            | Speech-to-text                           |
| **layoutLM**           | Document (PDF/image) understanding        |

---

# 🧪 Mini Project Ideas (Pick One)

1. 🗞️ **Smart News Summarizer**:
   - Use **t5-small** to summarize long news articles or blog posts.

2. 😊 **AI Resume Reviewer**:
   - Use **bert-base-uncased** to classify if a resume is suitable for a job description.

3. ❓ **Question Answering Assistant**:
   - Build a Q&A system using **distilbert-base-cased** on a custom FAQ or PDF content.

---

# ✅ Recap Checklist

| **Task**                                | **Status** |
|-----------------------------------------|------------|
| Understood transformer concepts         | ✅          |
| Used pre-trained models with pipeline   | ✅          |
| Worked with Hugging Face datasets       | ✅          |
| Loaded and tested a custom model        | ✅          |
| Built a mini-project using transformers | ✅          |

---

By completing these tasks, you’ve gained hands-on experience with Hugging Face Transformers and are now ready to tackle advanced NLP projects. 🚀