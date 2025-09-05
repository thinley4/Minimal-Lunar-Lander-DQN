# 1. Implement a basic Deep Q-Network (DQN) agent using Stable Baselines3 

# Why MlpPlicy?

# 1. Lunar Lander observations are numerical vectors (not images)
# 2. MLP can capture relationships between position, velocity, and 
# optimal actions


def dqn_agent(env, DQN):
    return DQN("MlpPolicy", env, verbose=1) 