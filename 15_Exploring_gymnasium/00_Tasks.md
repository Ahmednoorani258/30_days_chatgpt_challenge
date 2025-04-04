# 🚀 Day 15: Experiment with Gymnasium for Reinforcement Learning

Today, you will explore **Gymnasium**, the modern replacement for OpenAI Gym, and understand how reinforcement learning (RL) environments work. You’ll create a simple random agent to interact with an environment and analyze how agents learn through trial and error.

---

## 🎯 Goal

- Understand what Gymnasium is and why it replaces OpenAI Gym.
- Learn how RL environments work (observations, actions, rewards).
- Run a simple Gymnasium environment.
- Create a random agent that interacts with the environment.
- Understand the limitations of random actions in RL.

---

## 🔄 Why Gymnasium Instead of OpenAI Gym?

- 🚀 **Actively maintained & updated**.
- 🔥 **More features, better performance**.
- ✅ **Works well with modern RL libraries** (e.g., Stable-Baselines3).
- 💡 **Supports newer Python and ML frameworks**.

---

## 📌 Tasks for Day 15

### 🔹 Step 1: Install Gymnasium
First, install Gymnasium and additional dependencies:

```
pip install gymnasium
If you want to use environments with rendering (e.g., CartPole, Atari):

pip install gymnasium[classic_control]
For all environments:

pip install gymnasium[all]
```
## 🔹 Step 2: Understanding RL Environments

Each Gymnasium environment has the following components:

- **Observation Space**: The state information available to the agent.
- **Action Space**: The set of possible actions an agent can take.
- **Rewards**: The feedback received for each action.
- **Episode**: A sequence of actions until termination.

### Examples:
- **CartPole-v1**: Balance a pole on a moving cart.
- **MountainCar-v0**: Help a car reach the top of a hill.

---

## 🔹 Step 3: Running a Gymnasium Environment

### Tasks:
1. Load an environment (e.g., `CartPole-v1`).
2. Print its **state space** and **action space**.
3. Take random actions and observe how the agent interacts with the environment.

---

## 🔹 Step 4: Implement a Random Agent

A random agent selects actions randomly without learning. This helps you understand how environments work before applying machine learning.

---

## 🔹 Step 5: Run Multiple Episodes

### Tasks:
1. Run at least **10 episodes** where the agent takes random actions.
2. Track rewards per episode.
3. Observe how long the agent survives in each episode.

---

## 🔹 Step 6: Visualize the Agent’s Performance

- Use Gymnasium's **render mode** to see the agent’s actions in real-time.

---

## 🔹 Step 7: Analyze Performance

- Observe why a random agent performs poorly.
- Think about how a trained RL model could improve performance.

---

## 📊 What You Should Document Today

1. ✅ The environment you used (e.g., `CartPole-v1`).
2. ✅ The **observation space**, **action space**, and **reward system**.
3. ✅ How your random agent performed.
4. ✅ Why reinforcement learning is necessary for better performance.