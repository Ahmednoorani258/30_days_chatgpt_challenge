# PyTorch libraries import karte hain jo model banane ke liye zaroori hain
import torch

# Neural networks banane ke liye module import karte hain
import torch.nn as nn

# Optimizer ke liye module import karte hain jo model ko train karta hai
import torch.optim as optim

# MNIST dataset aur transforms ke liye torchvision se import karte hain
from torchvision import datasets, transforms

# Data ko batches mein load karne ke liye DataLoader import karte hain
from torch.utils.data import DataLoader

# Generated images ko dikhane ke liye matplotlib import karte hain
import matplotlib.pyplot as plt

# Random seed set karte hain taake har baar same results milen
torch.manual_seed(42)

# Images ko tensor mein convert karne ka transform define karte hain
transform = transforms.Compose([transforms.ToTensor()])
# MNIST dataset download aur load karte hain (train=True matlab training data)
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
# DataLoader banate hain jo data ko 128 ke batches mein deta hai aur shuffle karta hai
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# VAE class define karte hain jo model ka structure banayega
class VAE(nn.Module):
    # Constructor function define karte hain
    def __init__(self):
        # Parent class ko initialize karte hain
        super(VAE, self).__init__()
        # Encoder define karte hain jo image ko chhota code banayega
        self.encoder = nn.Sequential(
            # Pehla layer: 784 pixels (28x28) se 256 nodes tak
            nn.Linear(784, 256),
            # ReLU activation function jo negative values ko zero karta hai
            nn.ReLU(),
            # Dusra layer: 256 nodes se 32 nodes tak (16 mean + 16 log variance)
            nn.Linear(256, 32),
        )
        # Decoder define karte hain jo code se wapas image banayega
        self.decoder = nn.Sequential(
            # Pehla layer: 16 nodes se 256 nodes tak
            nn.Linear(16, 256),
            # ReLU activation function
            nn.ReLU(),
            # Dusra layer: 256 nodes se 784 pixels tak
            nn.Linear(256, 784),
            # Sigmoid function jo output ko 0 se 1 ke beech rakhta hai
            nn.Sigmoid(),
        )

    # Reparameterize function jo random sampling karta hai latent space mein
    def reparameterize(self, mu, logvar):
        # Standard deviation calculate karte hain log variance se
        std = torch.exp(0.5 * logvar)
        # Random noise generate karte hain jo std ke shape ka hai
        eps = torch.randn_like(std)
        # Mu aur noise ko combine karke latent vector return karte hain
        return mu + eps * std

    # Forward function jo pura model chalata hai
    def forward(self, x):
        # Image ko flatten karte hain (28x28 = 784)
        x = x.view(-1, 784)
        # Encoder se code banate hain
        h = self.encoder(x)
        # Code ko mu (mean) aur logvar (log variance) mein baant dete hain
        mu, logvar = h[:, :16], h[:, 16:]
        # Reparameterize se latent vector banate hain
        z = self.reparameterize(mu, logvar)
        # Decoder se reconstructed image aur mu, logvar return karte hain
        return self.decoder(z), mu, logvar


# Loss function define karte hain jo model ko train karne mein madad karega
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss calculate karte hain (original aur reconstructed image ka difference)
    recon_loss = nn.BCELoss(reduction="sum")(recon_x, x.view(-1, 784))
    # KL divergence loss jo latent space ko regularize karta hai
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Dono losses ko combine karke total loss return karte hain
    return recon_loss + kl_loss


# Model ka object banate hain
model = VAE()
# Optimizer banate hain jo model ke parameters ko update karega
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training ke liye epochs ki tadaad set karte hain
num_epochs = 10
# Har epoch ke liye loop chalate hain
for epoch in range(num_epochs):
    # Model ko training mode mein set karte hain
    model.train()
    # Total loss ko track karne ke liye variable
    total_loss = 0
    # Har batch ke liye loop chalate hain
    for images, _ in train_loader:
        # Optimizer ke gradients ko zero karte hain
        optimizer.zero_grad()
        # Model se reconstructed images, mu, aur logvar nikalte hain
        recon_images, mu, logvar = model(images)
        # Loss calculate karte hain
        loss = vae_loss(recon_images, images, mu, logvar)
        # Backpropagation se gradients calculate karte hain
        loss.backward()
        # Optimizer step se model ke parameters update karte hain
        optimizer.step()
        # Batch ka loss total loss mein jodte hain
        total_loss += loss.item()
    # Epoch ke baad average loss print karte hain
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# Model ko evaluation mode mein set karte hain
model.eval()
# Gradient calculation band karte hain kyunki ab training nahi hai
with torch.no_grad():
    # 25 random latent vectors generate karte hain
    z = torch.randn(25, 16)
    # Decoder se naye images banate hain aur unhe 28x28 shape dete hain
    generated_images = model.decoder(z).view(-1, 1, 28, 28)

# Generated images ko 5x5 grid mein dikhane ke liye plot banate hain
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# Har image ke liye loop chalate hain
for i, img in enumerate(generated_images):
    # Grid mein sahi position select karte hain
    ax = axes[i // 5, i % 5]
    # Image ko grayscale mein dikhate hain
    ax.imshow(img.squeeze(), cmap="gray")
    # Axis ko hide karte hain taake saaf dikhe
    ax.axis("off")
# Images ko file mein save karte hain
plt.savefig("generated_digits.png")
# Images ko screen pe dikhate hain
plt.show()
