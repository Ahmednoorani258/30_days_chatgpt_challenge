# 💡 Background:
# CartPole-v1 is a classic Reinforcement Learning problem where:
# - A cart moves on a track (left or right).
# - A pole is attached to the cart.
# - The goal is to take actions to keep the pole balanced and prevent it from falling.

import gymnasium as gym

# Load the CartPole-v1 environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to get the initial state
state, info = env.reset()

# ✅ 1. Observation Space
print("Observation Space:", env.observation_space)
# Observation Space: Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf], (4,), float32)
# Explanation:
# The agent observes 4 features from the environment:
# 1. Cart Position: Where the cart is on the track (left/right).
# 2. Cart Velocity: How fast the cart is moving.
# 3. Pole Angle: How tilted the pole is (left/right).
# 4. Pole Angular Velocity: How fast the pole is tilting.

# ✅ 2. Action Space
print("Action Space:", env.action_space)
# Action Space: Discrete(2)
# Explanation:
# The agent has 2 possible actions:
# 0 = Move Left
# 1 = Move Right

# ✅ 3. Initial State
print("Initial State:", state)
# Initial State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
# Example:
# Cart Position: 0.0036 (near the center)
# Cart Velocity: -0.0056 (slightly moving left)
# Pole Angle: -0.0056 (slightly tilted left)
# Pole Angular Velocity: -0.0415 (slowly rotating left)

# ✅ 4. Action
action = env.action_space.sample()  # Randomly pick an action (0 or 1)
print("Action Taken:", action)

# ✅ 5. Next State
next_state, reward, done, truncated, info = env.step(action)
print("Next State:", next_state)
# Example Next State:
# Cart Position: 0.00349 (cart shifted slightly left)
# Cart Velocity: -0.2006 (cart moving left faster)
# Pole Angle: -0.00649 (pole tilted slightly more left)
# Pole Angular Velocity: 0.2493 (pole now rotating slightly right)

# ✅ 6. Reward
print("Reward:", reward)
# Reward: 1.0
# Explanation:
# The agent gets +1 reward for every successful move where the pole doesn't fall.

# ✅ 7. Done
print("Done:", done)
# Done: False
# Explanation:
# If the pole falls or the cart moves out of bounds, `done = True`. Here, it's False, so the game continues.

# ✅ 8. Truncated (optional)
print("Truncated:", truncated)
# Truncated: False
# Explanation:
# If the episode ends due to a time limit (not because of failure), `truncated = True`.

# 📊 Visualization (Conceptual Representation):
# Track:  ---------------------------------------
# Cart:                   [C]
# Pole:                    |

# After one move:
# Cart:               [C]
# Pole:                 /

# Observation: The cart moved slightly left, and the pole tilted more.

# 🧠 Summary:
# Step-by-step process:
# 1. Observation: The agent observes 4 features from the environment.
# 2. Action: The agent takes a random action (left or right).
# 3. Step: The environment updates, and the agent receives a new state.
# 4. Reward: The agent gets +1 reward if the pole doesn't fall.
# 5. Done: The environment checks if the game is over.

# Close the environment after use
env.close()