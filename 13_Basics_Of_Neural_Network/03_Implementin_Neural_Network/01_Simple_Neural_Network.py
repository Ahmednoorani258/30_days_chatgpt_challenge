# Import necessary libraries
import tensorflow as tf           # TensorFlow is a popular deep learning library.
from tensorflow import keras      # Keras is a high-level API for building and training neural networks.
import numpy as np                # NumPy is used for handling numerical data (arrays).

# -----------------------------------------------
# Step 1: Define the Neural Network Architecture
# -----------------------------------------------
# We use a Sequential model, which means we build the network layer by layer.

model = keras.Sequential([
    # Hidden Layer:
    # - Dense layer with 2 neurons.
    # - 'relu' activation function (Rectified Linear Unit) introduces non-linearity.
    # - input_shape=(2,) means each input has 2 features (for XOR, each input is a pair [x1, x2]).
    keras.layers.Dense(2, activation="relu", input_shape=(2,)),
    
    # Output Layer:
    # - Dense layer with 1 neuron.
    # - 'sigmoid' activation function squashes the output to a range between 0 and 1,
    #   making it suitable for binary classification.
    keras.layers.Dense(1, activation="sigmoid")
])

# -----------------------------------------------
# Step 2: Compile the Model
# -----------------------------------------------
# - 'optimizer="adam"': Adam is an adaptive learning rate optimizer.
# - 'loss="binary_crossentropy"': Binary crossentropy is used for binary classification problems.
# - 'metrics=["accuracy"]': We track the accuracy during training.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# -----------------------------------------------
# Step 3: Prepare the XOR Training Data
# -----------------------------------------------
# XOR (exclusive OR) is a classic problem where:
# - Input: [0, 0] -> Output: 0
# - Input: [0, 1] -> Output: 1
# - Input: [1, 0] -> Output: 1
# - Input: [1, 1] -> Output: 0
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
              
# The target outputs for the XOR problem.
y = np.array([[0],
              [1],
              [1],
              [0]])

# -----------------------------------------------
# Step 4: Train the Model
# -----------------------------------------------
# We train the model using the training data for 500 epochs.
# 'epochs=500' means the entire dataset is passed through the network 500 times.
# 'verbose=0' silences the output during training.
model.fit(X, y, epochs=500, verbose=0)

# -----------------------------------------------
# Step 5: Test the Model
# -----------------------------------------------
# Use the trained model to predict the outputs for the input data.
predictions = model.predict(X)

# Round predictions to 0 or 1 to interpret them as binary outputs.
print("Predictions:", predictions.round())

# Explanation:
# - The network learns the XOR function by adjusting weights and biases over multiple epochs.
# - Although XOR is not linearly separable, a network with a hidden layer (with 'relu' activation) can solve it.
# - The final predictions printed should be close to the desired XOR outputs: [0, 1, 1, 0].


# output
# Predictions: [[0.]
#  [1.]
#  [1.]
#  [0.]]
#