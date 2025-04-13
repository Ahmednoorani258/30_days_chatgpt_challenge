import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and prepare MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 32),  # Latent space (16 mean + 16 log variance)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten image
        # Encode
        h = self.encoder(x)
        mu, logvar = h[:, :16], h[:, 16:]
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        return self.decoder(z), mu, logvar


# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction="sum")(recon_x, x.view(-1, 784))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# Initialize model, optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# Generate new digits
model.eval()
with torch.no_grad():
    z = torch.randn(25, 16)  # Random codes for 25 images
    generated_images = model.decoder(z).view(-1, 1, 28, 28)

# Show generated digits
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, img in enumerate(generated_images):
    ax = axes[i // 5, i % 5]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.axis("off")
plt.savefig("generated_digits.png")
plt.show()
