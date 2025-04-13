import gymnasium as gym
import streamlit as st

st.title("Gymnasium Environment Explorer")
st.write("Exploring the CartPole-v1 environment")

# Create the environment
env = gym.make("CartPole-v1")

# Display environment details
st.write("**Observation Space:**", env.observation_space)
st.write("**Action Space:**", env.action_space)
st.write("**Description:** Balance a pole on a cart by moving left or right.")

# Run random agent when button is clicked
if st.button("Run Random Agent"):
    state = env.reset()
    total_reward = 0
    done = False
    truncated = False
    while not (done or truncated):
        action = env.action_space.sample()
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    st.write(f"Total Reward: {total_reward}")
    env.close()
