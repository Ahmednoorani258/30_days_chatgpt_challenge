# 🚀 Day 20: Dive into Generative AI Basics

## 🎯 Goal
Gain a solid understanding of **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)**—two foundational generative models. By the end of today, you’ll understand their architectures, training procedures, and practical applications, and you’ll experiment with pre-built implementations.

---

## 📚 What You Will Learn
1. Core concepts behind **GANs** and **VAEs**.
2. The architecture of a **GAN** (Generator vs. Discriminator).
3. The architecture of a **VAE** (Encoder vs. Decoder, Latent Space).
4. Training dynamics:
   - **GANs**: Adversarial training (Generator vs. Discriminator).
   - **VAEs**: Reconstruction + KL Divergence Loss.
5. Practical use cases:
   - Image synthesis.
   - Data augmentation.
   - Anomaly detection.

---

## 🛠 Tools & Libraries
| **Tool**               | **Purpose**                                      |
|-------------------------|--------------------------------------------------|
| **PyTorch** or **TensorFlow/Keras** | Framework for building and training models. |
| **Torchvision** / **tf.keras.datasets** | Sample datasets (e.g., MNIST, CIFAR-10).   |
| **Matplotlib**          | Visualizations of generated images and losses.  |
| **Google Colab** (optional) | GPU acceleration for faster training.        |

---

## 🔧 Step-by-Step Tasks

### 1️⃣ Understand the Theory

#### GANs:
- Two networks:
  - **Generator**: Creates fake data.
  - **Discriminator**: Distinguishes real vs. fake data.
- **Adversarial Training**: The Generator and Discriminator compete, improving each other over time.

#### VAEs:
- **Encoder**: Maps input to a latent distribution.
- **Decoder**: Reconstructs input from latent samples.
- **Loss Function**: 
  - **Reconstruction Loss**: Measures how well the output matches the input.
  - **KL Divergence**: Pushes the latent distribution toward a prior (e.g., Gaussian).

📖 **Resources**:
- [Ian Goodfellow’s Original GAN Paper Overview](https://arxiv.org/abs/1406.2661)
- Blog posts:
  - [“GANs in 50 Lines of Code”](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-f6e9907a0283)
  - [“Understanding VAEs”](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

---

### 2️⃣ Explore Pre-Built Implementations

#### PyTorch Examples:
- [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [VAE Tutorial](https://pytorch.org/tutorials/beginner/vae.html)

#### TensorFlow/Keras Examples:
- [tf.keras GAN Example](https://www.tensorflow.org/tutorials/generative/dcgan)
- [tf.keras VAE Example](https://www.tensorflow.org/tutorials/generative/cvae)

---

### 3️⃣ Hands-On: Generate Images with a Pretrained GAN
1. Load a pretrained **DCGAN** (e.g., trained on MNIST or CelebA) from PyTorch’s model zoo or Hugging Face.
2. Generate samples by feeding random noise vectors to the Generator.
3. Visualize a grid of generated images using **Matplotlib**.

---

### 4️⃣ Hands-On: Encode & Decode with a Pretrained VAE
1. Load a pretrained **VAE** (e.g., trained on MNIST).
2. Encode a batch of real images into latent vectors.
3. Decode latent vectors back into images.
4. Interpolate between two latent vectors and visualize the smooth transition in generated images.

---

### 5️⃣ Analyze Training Dynamics

#### GAN:
- Plot **Generator** and **Discriminator** losses over epochs to observe the adversarial “game.”

#### VAE:
- Plot **Reconstruction Loss** vs. **KL Divergence Loss** to see the balance between the two components.

---

### 6️⃣ Mini Project Ideas

1. **Data Augmentation with GANs**:
   - Use a GAN to generate additional samples for an imbalanced dataset.

2. **Anomaly Detection with VAE**:
   - Train a VAE on “normal” data and detect anomalies by measuring high reconstruction error.

3. **Latent Space Exploration**:
   - Build an interactive tool (e.g., using **Streamlit**) to manipulate latent vectors and generate new images.

---

## ✅ Day 20 Checklist

| **Task**                                                                 | **Status** |
|--------------------------------------------------------------------------|------------|
| Explained GAN and VAE architectures and loss functions                   | ✅          |
| Explored official tutorials/examples for both models                     | ✅          |
| Generated images using a pretrained GAN and visualized them              | ✅          |
| Encoded/decoded images with a pretrained VAE; performed latent interpolation | ✅      |
| Analyzed training losses for both GAN and VAE                            | ✅          |
| Chosen one mini project idea to implement next                           | ✅          |

---

By completing these tasks, you’ve gained hands-on experience with **GANs** and **VAEs**, two of the most powerful generative AI models. You’re now ready to apply these techniques to real-world projects! 🚀