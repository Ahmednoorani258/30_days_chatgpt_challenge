# 🚀 Day 21: Mini Generative AI Project

## 🎯 Goal
Use a **Generative AI model** (GAN or VAE) to build a mini project that showcases your ability to train, generate, and visualize new data (like images or audio). This project will solidify your understanding of model architecture, training, inference, and creativity.

---

## 🧠 What You Will Learn
1. Applying **GANs/VAEs** on real datasets.
2. Dataset preprocessing for generative models.
3. Training a generator and evaluating output quality.
4. Visual storytelling of generative results.
5. (Optional) Deploying your model with **Streamlit**.

---

## 🛠️ Tools & Libraries

| **Tool**               | **Use**                                   |
|-------------------------|-------------------------------------------|
| **PyTorch / TensorFlow** | Model training                           |
| **torchvision / keras.datasets** | Datasets (e.g., MNIST, CIFAR-10, CelebA) |
| **Matplotlib / Seaborn** | Image visualization                      |
| **Streamlit** (optional) | Simple UI for demo                       |
| **Weights & Biases** (optional) | Experiment tracking                |

---

## 🧩 Step-by-Step Tasks

### ✅ 1. Choose a Project Idea
Here are a few beginner-friendly but impactful projects you can build in 1–2 days:

| **Project**                  | **Description**                                                                 |
|------------------------------|---------------------------------------------------------------------------------|
| 🎨 **Handwritten Digit Generator** | Train a DCGAN on MNIST and generate new digits.                              |
| 🧠 **Latent Space Explorer**       | Train a VAE on MNIST/CIFAR-10 and create an app to explore latent space.      |
| 😍 **Anime Face Generator**        | Use a pre-trained StyleGAN (or fine-tune) to generate anime faces.           |
| 🚨 **Anomaly Detector (VAE)**      | Train a VAE to detect abnormal samples via reconstruction error.             |
| 🎶 **Music Note Generator** (Advanced) | Use MIDI datasets and RNN/VAE to generate music.                          |

---

### ✅ 2. Prepare the Dataset
Use existing datasets like **MNIST**, **FashionMNIST**, **CIFAR-10**, or custom datasets (e.g., small image folders).

- Normalize images to `[-1, 1]` if using GANs (use `transforms.Normalize`).
- Resize images to `64x64` or `128x128` (depending on your model size).

#### Example: Load and Preprocess MNIST Dataset
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='./data', transform=transform, download=True)
```
✅ 3. Train the Model (GAN or VAE)
Start with a simple GAN or VAE using a DCGAN or standard VAE architecture.
Use a training loop for at least 5–20 epochs.
Plot loss curves every few epochs.
🧠 Training Tips:
Use BatchNorm in the generator and discriminator.
Keep the generator and discriminator balanced—avoid one overpowering the other.
✅ 4. Generate and Visualize New Samples
Save model checkpoints.
Generate samples from latent vectors (noise in GANs or encoded latent space in VAEs).
Use Matplotlib to create grids of outputs.
Example: Visualize Generated Images
Use matplotlib to create grids of outputs.

```python

import matplotlib.pyplot as plt

def show_images(images, n_cols=5):
    fig, axes = plt.subplots(len(images) // n_cols, n_cols, figsize=(10, 10))
    for i, img in enumerate(images):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.show()
```
✅ 5. (Optional) Build a Simple Streamlit UI
Show off your project with a small web interface:

```python
pip install streamlit
Example:

import streamlit as st

st.title("GAN Image Generator")
z = st.slider("Latent Vector Seed", 0, 100)
# generate image based on latent vector and show
st.image(generated_image, caption="Generated Sample")
```
# ✅ Write a Short Report

Document your project in Markdown or Notion. Include the following:

1. **Dataset Used**:
   - Mention the dataset and preprocessing steps.

2. **Model Architecture**:
   - Describe the generator, discriminator, or encoder-decoder.

3. **Loss Curves**:
   - Include plots of training losses (e.g., Generator vs. Discriminator loss for GANs or Reconstruction vs. KL Divergence loss for VAEs).

4. **Sample Generations**:
   - Show generated outputs (e.g., images, audio, or other data).

5. **What Worked and What Didn’t**:
   - Reflect on challenges, successes, and observations during the project.

6. **Future Improvements**:
   - Suggest what you’d do differently next time or additional features you’d like to add.

This report will help you prepare for portfolio showcasing or interview discussions.

---

## 🎁 Bonus Ideas (Push Further)

1. **Train on Your Own Dataset**:
   - Use custom data, such as your drawings, face dataset, or any other unique dataset.

2. **Interactive Sliders**:
   - Add sliders to manipulate latent dimensions (e.g., StyleGAN demos).

3. **Training Visualization**:
   - Use **Weights & Biases (W&B)** or **TensorBoard** to track and visualize training logs.

---

## ✅ Day 21 Checklist

| **Task**                                | **Done?** |
|-----------------------------------------|-----------|
| Chose a Generative AI project           | ✅         |
| Loaded + preprocessed dataset           | ✅         |
| Trained model (GAN/VAE)                 | ✅         |
| Generated and visualized new data       | ✅         |
| (Optional) Built a Streamlit demo       | ✅         |
| Documented findings and learnings       | ✅         |

---

By completing this report and exploring bonus ideas, you’ll have a polished project ready for your portfolio and a deeper understanding of Generative AI. 🚀