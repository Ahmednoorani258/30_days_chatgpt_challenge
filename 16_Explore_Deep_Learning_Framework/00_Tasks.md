# 🚀 Day 16: Explore Deep Learning Frameworks (PyTorch & TensorFlow)

## 🎯 Goal
Get introduced to the two most popular deep learning frameworks — **PyTorch** and **TensorFlow**. By the end of the day, you'll understand the core concepts, syntax differences, and how to build a simple neural network with each.

---

## 🧠 Why This Matters
In AI agent development, deep learning is the brain behind powerful models. Whether you're training an image classifier or an agent to play games, understanding frameworks like PyTorch and TensorFlow is essential.

---

## 📚 What You Will Learn
- ✅ Difference between **PyTorch** and **TensorFlow**.
- ✅ How to install and configure both frameworks.
- ✅ Build and train a basic neural network using both.
- ✅ Learn key concepts like **Tensors**, **Autograd**, and **Model Training Loop**.
- ✅ Best use cases for each framework.

---

## 🧰 Tools You'll Use Today
- ✅ **PyTorch**
- ✅ **TensorFlow**
- ✅ **Jupyter Notebook** or **VS Code**
- ✅ **GPU/Colab** (if needed)

---

## 🔧 Step-by-Step Tasks

### 🔹 Step 1: Install the Frameworks
Install PyTorch and TensorFlow using the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow
```
## 💡 If Using GPU
- Check PyTorch and TensorFlow's official sites for GPU setup instructions to leverage faster training.

---

## 🔹 Step 2: Understand Tensors

### What is a Tensor?
- A **Tensor** is a multi-dimensional array, the fundamental building block of deep learning frameworks.

### Tasks:
1. Create tensors in both **PyTorch** and **TensorFlow**.
2. Perform operations like addition, multiplication, and reshaping.
3. Compare how tensors are handled in PyTorch vs TensorFlow.

---

## 🔹 Step 3: Build a Basic Neural Network in PyTorch

### Goal:
- Build a model to classify digits (e.g., MNIST-like dataset).

### Tasks:
1. Define the architecture using `nn.Module`.
2. Implement a training loop with an optimizer and loss function.
3. Log metrics like **loss** and **accuracy** during training.

---

## 🔹 Step 4: Build a Similar Model in TensorFlow/Keras

### Goal:
- Build the same model using TensorFlow/Keras.

### Tasks:
1. Use the **Sequential** or **Functional API** to define the model.
2. Train the model using the `.fit()` method.
3. Compare the syntax and simplicity with PyTorch.

---

## 🔹 Step 5: Analyze the Differences

### Comparison Table:

| **Feature**       | **PyTorch**                  | **TensorFlow**              |
|--------------------|------------------------------|-----------------------------|
| **Coding Style**   | Pythonic, Flexible          | More structured, High-level |
| **Debugging**      | Easier (Dynamic Graphs)     | More abstract               |
| **Community**      | Strong in Research          | Strong in Industry          |
| **Popular API**    | `torch.nn`, `autograd`      | `tf.keras`, `GradientTape`  |

---

## ✅ Deliverables for Day 16

1. A simple neural network implemented in **PyTorch**.
2. A similar neural network implemented in **TensorFlow**.
3. Documentation on which framework felt easier and why.
4. Save training logs or screenshots of performance.

---

## 💡 Bonus

1. Try training on a real dataset like **Iris** or **Fashion-MNIST**.
2. If you're using **Google Colab**, enable GPU for faster training.