# 🚀 Day 15: Experiment with OpenAI Gym

Today, you will explore **OpenAI Gym**, a popular toolkit for developing and testing Reinforcement Learning (RL) algorithms. You will set up an RL environment, interact with it, and implement a basic agent.

---

## 🎯 Goal

1. Understand the core concepts of OpenAI Gym.
2. Learn how agents interact with an environment.
3. Implement a random agent and analyze its performance.
4. Prepare for future reinforcement learning models.

---

## 📚 Topics to Cover

### ✅ 1. What is OpenAI Gym?
- OpenAI Gym is a library that provides prebuilt environments for training RL models.
- It includes environments like **CartPole**, **MountainCar**, **Atari games**, and more.

### ✅ 2. Understanding Gym Environments
Each RL environment in Gym follows this structure:
- **State (Observation Space)**: The agent’s perception of the environment.
- **Action Space**: The set of actions an agent can take.
- **Rewards**: The feedback given based on the agent’s actions.
- **Episode**: A full sequence of actions until a termination condition is met.

### ✅ 3. Installing OpenAI Gym
Before using Gym, you need to install it. Run:
```bash
pip install gym
```
For advanced environments like Atari games:
```
pip install gym[atari]
```

## ✅ 4. Running a Prebuilt Gym Environment

OpenAI Gym provides simple environments like **CartPole**, where a pole is balanced on a moving cart.

### You Will:
- Load the environment.
- Observe its states.
- Take random actions.

---

## ✅ 5. Exploring the State and Action Space

### Tasks:
- Print the **observation space** (state information).
- Print the **action space** (possible actions the agent can take).

---

## ✅ 6. Implementing a Random Agent

A random agent selects actions randomly without learning. This helps us understand how Gym works before training an RL model.

### You Will:
- Run multiple episodes.
- Take random actions at each step.
- Observe rewards and episode termination.

---

## ✅ 7. Visualizing the Agent’s Performance

### Tasks:
- Render the environment to see the agent's behavior.
- Analyze performance metrics (e.g., average reward per episode).

---

## ✅ 8. Analyzing Results

### Tasks:
- Observe why random actions do not perform well.
- Discuss how learning-based approaches can improve performance.

---

## 🛠 Tasks for Day 15
🔹 Step 1: Install OpenAI Gym
If you haven’t installed Gym, do it now:


pip install gym
🔹 Step 2: Load and Explore an Environment
Load CartPole-v1 or MountainCar-v0.

Print the state and action space.

🔹 Step 3: Implement a Random Agent
Write a Python script where an agent selects actions randomly.

🔹 Step 4: Run Multiple Episodes
Let the agent run for 1000 steps per episode and record the rewards.

🔹 Step 5: Visualize the Agent’s Performance
Render the environment to see the agent's decisions.

🔹 Step 6: Analyze Performance
Compare episodes to see if random actions achieve good results.

🔹 Step 7: Document Findings
Write down what you learned about state space, actions, and rewards.

