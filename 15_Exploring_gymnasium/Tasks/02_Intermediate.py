import gymnasium as gym
import streamlit as st
import pandas as pd

st.title("Multi-Environment Random Agent Dashboard")

# Environment selection
env_options = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"]
env_name = st.selectbox("Select Environment", env_options)
env = gym.make(env_name)

# Display details
st.write("**Observation Space:**", env.observation_space)
st.write("**Action Space:**", env.action_space)

# Number of episodes
num_episodes = st.slider("Number of Episodes", 1, 50, 5)

# Run agent when button clicked
if st.button("Run Random Agent"):
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    env.close()

    st.write(f"Average Reward: {sum(rewards) / num_episodes}")
    # Plot rewards
    df = pd.DataFrame({"Episode": range(1, num_episodes + 1), "Reward": rewards})
    st.bar_chart(df.set_index("Episode"))
