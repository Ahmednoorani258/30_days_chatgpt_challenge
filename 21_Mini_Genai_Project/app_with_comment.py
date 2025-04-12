# PyTorch aur torchvision libraries import karte hain
# PyTorch deep learning ke liye aur torchvision datasets aur transforms ke liye use hoti hai
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform images ko preprocess karte hain taake model unhe samajh sake
# ToTensor() images ko tensors mein convert karta hai
# Normalize() pixel values ko [-1, 1] range mein laata hai
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# MNIST dataset download aur load karte hain
# MNIST handwritten digits ka dataset hai jo 28x28 grayscale images ka collection hai
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# DataLoader ka kaam hai dataset ko batches mein divide karna aur shuffle karna
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# PyTorch ka nn module import karte hain jo neural networks banane ke liye use hota hai
import torch.nn as nn

# Generator class define karte hain jo random noise ko images mein convert karega
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Sequential layers define karte hain
        self.model = nn.Sequential(
            nn.Linear(100, 256),  # Input: 100 random numbers (latent vector)
            nn.ReLU(),            # Activation function: ReLU
            nn.Linear(256, 512),  # Hidden layer
            nn.ReLU(),
            nn.Linear(512, 1024), # Hidden layer
            nn.ReLU(),
            nn.Linear(1024, 28*28),  # Output: 784 pixels (28x28 image)
            nn.Tanh()               # Normalize output to [-1, 1]
        )
    
    # Forward function: input ko process karna
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # Output ko 28x28 image mein reshape karte hain

# Discriminator class define karte hain jo real aur fake images ko differentiate karega
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Sequential layers define karte hain
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),    # Input: Flattened 28x28 image (784 pixels)
            nn.LeakyReLU(0.2),        # Activation function: LeakyReLU
            nn.Linear(512, 256),      # Hidden layer
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),        # Output: Single value (real or fake)
            nn.Sigmoid()              # Normalize output to [0, 1]
        )
    
    # Forward function: input ko process karna
    def forward(self, img):
        img_flat = img.view(-1, 28*28)  # Image ko flatten karte hain
        return self.model(img_flat)

# Generator aur Discriminator ke objects banate hain
generator = Generator()
discriminator = Discriminator()

# Optimizers aur loss function import karte hain
import torch.optim as optim

# Loss function: Binary Cross Entropy Loss (BCE Loss)
criterion = nn.BCELoss()

# Optimizers: Adam optimizer use karte hain learning ke liye
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)  # Generator ke liye
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)  # Discriminator ke liye

# Training ke liye number of epochs define karte hain
num_epochs = 10
for epoch in range(num_epochs):
    for real_images, _ in train_loader:  # Real images ko DataLoader se load karte hain
        batch_size = real_images.size(0)  # Batch size nikalte hain
        
        # Real aur fake labels define karte hain
        real_labels = torch.ones(batch_size, 1)  # Real images ke liye label = 1
        fake_labels = torch.zeros(batch_size, 1)  # Fake images ke liye label = 0
        
        # Discriminator ko train karte hain
        optimizer_d.zero_grad()  # Gradients ko zero karte hain
        
        # Real images ka loss calculate karte hain
        outputs = discriminator(real_images)  # Real images ko Discriminator mein pass karte hain
        d_loss_real = criterion(outputs, real_labels)  # Real loss calculate karte hain
        
        # Fake images ka loss calculate karte hain
        noise = torch.randn(batch_size, 100)  # Random noise generate karte hain
        fake_images = generator(noise)  # Noise ko Generator mein pass karte hain
        outputs = discriminator(fake_images.detach())  # Fake images ko Discriminator mein pass karte hain
        d_loss_fake = criterion(outputs, fake_labels)  # Fake loss calculate karte hain
        
        # Total Discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()  # Backpropagation
        optimizer_d.step()  # Discriminator ko update karte hain
        
        # Generator ko train karte hain
        optimizer_g.zero_grad()  # Gradients ko zero karte hain
        outputs = discriminator(fake_images)  # Fake images ko Discriminator mein pass karte hain
        g_loss = criterion(outputs, real_labels)  # Generator ka loss (Discriminator ko fool karna)
        g_loss.backward()  # Backpropagation
        optimizer_g.step()  # Generator ko update karte hain
    
    # Har epoch ke baad loss print karte hain
    print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Matplotlib import karte hain images ko visualize karne ke liye
import matplotlib.pyplot as plt

# Random noise generate karte hain aur images banate hain
noise = torch.randn(25, 100)  # 25 random codes
generated_images = generator(noise)  # Generator se images banate hain

# Images ko 5x5 grid mein plot karte hain
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, img in enumerate(generated_images):
    ax = axes[i // 5, i % 5]
    ax.imshow(img.squeeze(), cmap='gray')  # Image ko grayscale mein dikhate hain
    ax.axis('off')  # Axis ko hide karte hain
plt.show()  # Images ko display karte hain