# Import necessary modules from the Hugging Face Transformers library and PyTorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Step 1: Load the tokenizer and model
# The tokenizer converts raw text into token IDs that the model can understand.
# The model is a pre-trained DistilBERT fine-tuned for sentiment analysis (binary classification: positive/negative).
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Step 2: Tokenize the input text
# The tokenizer processes the input text and returns a dictionary containing:
# - 'input_ids': Encoded token IDs for the input text.
# - 'attention_mask': A mask to distinguish real tokens from padding (useful for batching).
inputs = tokenizer(
    "AI is not changing the world!", return_tensors="pt"
)  # 'pt' specifies PyTorch tensors.

# Step 3: Pass the tokenized input to the model
# The model processes the input and outputs logits (raw scores for each class).
# Logits are unnormalized scores that indicate the model's confidence for each class.
# Print the tokenized inputs for debugging.
outputs = model(**inputs)  # The **inputs unpacks the dictionary into keyword arguments.
logits = outputs.logits  # Extract the logits from the model's output.

# Step 4: Convert logits to probabilities
# Apply the softmax function to the logits to normalize them into probabilities.
# Softmax ensures that the scores for all classes sum to 1, making them interpretable as probabilities.
probabilities = torch.softmax(
    logits, dim=1
)  # dim=1 applies softmax across the class dimension.

# Step 5: Determine the predicted class
# Use argmax to find the index of the class with the highest probability.
# The index corresponds to the predicted class (0 = NEGATIVE, 1 = POSITIVE).
predicted_class = torch.argmax(
    probabilities
).item()  # Convert the tensor to a Python integer.

# Step 6: Interpret the result
# Define the class labels for interpretation.
labels = ["NEGATIVE", "POSITIVE"]
# Use the predicted class index to retrieve the corresponding label.
print(f"Predicted sentiment: {labels[predicted_class]}")

# --- Step-by-Step Explanation ---
# 1. Tokenizer:
#    - Converts the input text "AI is changing the world!" into token IDs and an attention mask.
#    - Example: "AI is changing the world!" → [101, 993, 2003, 5279, 1996, 2088, 102]
# 2. Model:
#    - Processes the tokenized input and outputs logits (raw scores for each class).
#    - Example logits: tensor([[-2.3, 2.5]]) (negative score: -2.3, positive score: 2.5).
# 3. Softmax:
#    - Converts logits into probabilities.
#    - Example probabilities: tensor([[0.01, 0.99]]) (1% negative, 99% positive).
# 4. Argmax:
#    - Finds the class with the highest probability (index 1 = positive).
# 5. Interpretation:
#    - Maps the predicted class index to a human-readable label ("POSITIVE").

# --- Why Use This Approach? ---
# - Flexibility: You can fine-tune the model, adjust inputs, or integrate it into a larger system.
# - Customization: Unlike pipelines, this approach gives you full control over tokenization, model outputs, and post-processing.
# - Scalability: Ideal for batch processing or advanced use cases like multi-task learning.
