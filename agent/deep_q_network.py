# 1. Implement a basic Deep Q-Network (DQN) agent using Stable Baselines3 


from stable_baselines3 import DQN
from environment.lunar_lander_env import create_lunar_lander_env

env = create_lunar_lander_env()

def dqn_agent():
    return DQN("MlpPolicy", env, verbose=1) 