import gymnasium as gym

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# Start a new episode
state = env.reset()
done = False
truncated = False
total_reward = 0

# Run until the episode ends
while not (done or truncated):
    action = env.action_space.sample()  # Randomly pick 0 or 1
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Show the action in the window

print(f"Total Reward: {total_reward}")
env.close()


# _____________________________________________________
# _____________________________________________________
# Run Multiple agent (try this section by uncommenting)
# _____________________________________________________
# _____________________________________________________

# import gymnasium as gym

# # Create the environment
# env = gym.make("CartPole-v1", render_mode="human")

# # Run 10 episodes
# for episode in range(10):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     truncated = False
    
#     while not (done or truncated):
#         action = env.action_space.sample()
#         state, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#         env.render()
    
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# env.close()