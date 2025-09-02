# 1. Implement a basic Deep Q-Network (DQN) agent using Stable Baselines3 


def dqn_agent(env, DQN):
    return DQN("MlpPolicy", env, verbose=1) 