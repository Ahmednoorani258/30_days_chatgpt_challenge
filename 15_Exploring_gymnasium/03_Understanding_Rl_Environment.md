# 🌟 Understanding RL Environments

An RL environment is like a game with rules. Gymnasium environments have four key components:

---

## 1️⃣ Observation Space

### What It Is:
- The information the agent “sees” about the environment.

### Example (CartPole-v1):
- The agent observes a 4-number list:
  1. **Cart position** (x-coordinate).
  2. **Cart velocity** (how fast it’s moving).
  3. **Pole angle** (how tilted the pole is).
  4. **Pole angular velocity** (how fast the pole is tilting).

### Think of It As:
- The agent’s “eyes” or sensors.

---

## 2️⃣ Action Space

### What It Is:
- The possible moves the agent can make.

### Example (CartPole-v1):
- **0**: Move the cart left.
- **1**: Move the cart right.

### Think of It As:
- The agent’s “hands” or controls.

---

## 3️⃣ Rewards

### What It Is:
- Feedback the agent gets after taking an action.

### Example (CartPole-v1):
- The agent gets **+1** for every step the pole stays upright.

### Think of It As:
- A score telling the agent “good job” or “try again.”

---

## 4️⃣ Episode

### What It Is:
- One full round of the game, from start to finish.

### Example (CartPole-v1):
- An episode starts with the pole upright and ends when:
  - The pole falls.
  - Or after 500 steps.

### Think of It As:
- A single “playthrough” of the game.

---

## 🏋️ Example Environments

1. **CartPole-v1**: Balance a pole on a cart for as long as possible.
2. **MountainCar-v0**: Get a car to the top of a hill by pushing it back and forth.