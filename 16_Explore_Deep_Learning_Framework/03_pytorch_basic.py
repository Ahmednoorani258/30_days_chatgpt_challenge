### Installing Pytorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### What’s installed?
# torch: The main PyTorch library.
# torchvision: Tools for image data (e.g., datasets like MNIST).
# torchaudio: Tools for audio (not needed today, but included).


import torch

print(torch.__version__)


# What is PyTorch?
# PyTorch is a Python library for building and training neural networks. It’s developed by Facebook’s AI team and is great for deep learning because:

# Dynamic: You can change your model on the fly (unlike TensorFlow’s older static graphs).
# Simple: Feels like regular Python with NumPy, plus GPU power.
# Research-Friendly: Easy to tweak and debug.
# Analogy: Think of PyTorch as a LEGO set for AI—flexible pieces (tensors) you snap together to build cool models, with a manual (Python) you already know.


# Tensors – The Building Blocks
# _________________________________
# Tensors are like NumPy arrays but with extra power: they can run on GPUs for faster computation.

# Create a tensor
x = torch.tensor([1, 2, 3])  # 1D tensor
print(x)  # tensor([1, 2, 3])

# 2D tensor (like a matrix)
y = torch.tensor([[1, 2], [3, 4]])
print(y)  # tensor([[1, 2], [3, 4]])

# zeros or ones
z = torch.zeros(2, 3)  # 2x3 tensor of zeros
print(z)  # tensor([[0., 0., 0.], [0., 0., 0.]])'

# Random Values
r = torch.rand(2, 2)  # 2x2 tensor with random numbers (0 to 1)
print(r)  # E.g., tensor([[0.1234, 0.5678], [0.9012, 0.3456]])

# MAth

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = a + b  # Element-wise addition
print(c)  # tensor([4, 6])


# Analogy: Tensors are like clay—you mold them into shapes (1D lists, 2D grids) and use them to sculpt your neural network.

# _________________________________
# Building a Neural Network
# PyTorch uses torch.nn to create neural networks. Let’s build a simple one to classify MNIST digits (like we did with TensorFlow).
# _________________________________


import torch.nn as nn


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()  # Flatten 28x28 images to 784
        self.layer1 = nn.Linear(784, 128)  # Hidden layer: 784 inputs -> 128 neurons
        self.relu = nn.ReLU()  # Activation
        self.layer2 = nn.Linear(128, 10)  # Output layer: 128 -> 10 (digits 0-9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Create the model
model = SimpleNet()
print(model)

# Explanation
# Class: We define a network by subclassing nn.Module—PyTorch’s base class for models.
# __init__:
# nn.Flatten(): Turns 28x28 images into a 784-long vector.
# nn.Linear(in, out): Fully connected layer (like Dense in Keras). E.g., 784 inputs to 128 outputs.
# nn.ReLU(): Adds non-linearity (outputs 0 for negatives, keeps positives).
# forward: Defines how data flows through the layers—input to output.
# Output: 10 numbers (raw scores for each digit).
# Analogy: Building the network is like stacking LEGO bricks—each layer adds a new piece to process the data.


# _________________________________
# 2 BASIC NEURAL NETWORK AGAIN
# _________________________________

# Build a Basic Neural Network in PyTorch
# Let’s build a neural network to classify digits (like MNIST, where images are 28x28 pixels, flattened to 784 inputs, and outputs are 10 classes: 0-9).

# Step-by-Step
# Define the Model:

# Use nn.Module to create a custom class.
# Add layers: input (784), hidden (128), output (10).

# Optimizer and Loss:

# Optimizer: Updates weights (e.g., SGD).
# Loss: Measures error (e.g., CrossEntropyLoss for classification).

# Training Loop:

# Forward pass: Compute predictions.
# Loss: Compare predictions to targets.

# Backward pass: Compute gradients.
# Update: Adjust weights.

import torch  # PyTorch library ko import karte hain, jo deep learning ke liye hai
import torch.nn as nn  # Neural network module, isme layers banane ka samaan hai
import torch.optim as optim  # Optimizer module, jo model ko behtar banane mein madad karta hai


# Model define karte hain - ek chhota sa neural network
class SimpleNN(nn.Module):  # SimpleNN naam ka model banaya, nn.Module se inherit kiya
    def __init__(self):  # Constructor - yahan model ke parts set karte hain
        super(
            SimpleNN, self
        ).__init__()  # Parent class (nn.Module) ko initialize karte hain
        # Pehli layer: 784 inputs (28x28 image) se 128 neurons tak
        self.fc1 = nn.Linear(784, 128)  # Fully connected layer 1
        # Doosri layer: 128 neurons se 10 outputs (0-9 digits) tak
        self.fc2 = nn.Linear(128, 10)  # Fully connected layer 2

    def forward(self, x):  # Forward pass - data ka flow yahan define hota hai
        # Pehli layer se guzar kar ReLU activation lagate hain (negatives ko 0 bana deta hai)
        x = torch.relu(self.fc1(x))  # Activation function
        # Doosri layer se guzar kar output banate hain
        x = self.fc2(x)
        return x  # Final output return karte hain (10 scores)


# Setup karte hain
model = SimpleNN()  # Model ka object banaya
# Optimizer set karte hain - SGD (Stochastic Gradient Descent) use kar rahe hain
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Learning rate 0.01 rakhi
# Loss function set karte hain - classification ke liye CrossEntropyLoss perfect hai
loss_fn = nn.CrossEntropyLoss()

# Dummy data banate hain (asli data baad mein replace kar sakte hain)
# 64 samples, har sample mein 784 features (jaise 28x28 image flatten karke)
input_data = torch.randn(64, 784)  # Random input data
print(input_data.shape)  # (64, 784) - 64 samples, har ek mein 784 features
# 64 labels, har label 0 se 9 tak ka ek number
target = torch.randint(0, 10, (64,))  # Random target labels (0-9)

# Training loop shuru karte hain
for epoch in range(10):  # 10 baar poora data dekhega (10 epochs)
    optimizer.zero_grad()  # Purane gradients ko zero karte hain, taaki naye saaf calculate hon
    output = model(input_data)  # Forward pass - model se prediction nikalte hain
    # Loss calculate karte hain - prediction aur asli labels mein kitna farq hai
    loss = loss_fn(output, target)
    loss.backward()  # Backward pass - loss se gradients calculate karte hain
    optimizer.step()  # Weights ko update karte hain taaki loss kam ho
    # Har epoch ke baad loss print karte hain taaki progress dikhe
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Explanation
# Model: Two layers—fc1 maps 784 inputs to 128 hidden units, fc2 maps 128 to 10 outputs. ReLU adds nonlinearity.
# Optimizer: SGD (Stochastic Gradient Descent) adjusts weights.
# Loss: CrossEntropyLoss combines softmax and loss calculation.
# Loop: Runs 10 times, updating weights each epoch.
# Output: You’ll see the loss decrease as the model learns!
