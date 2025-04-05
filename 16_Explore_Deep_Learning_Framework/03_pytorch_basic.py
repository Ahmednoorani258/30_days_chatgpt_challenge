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

