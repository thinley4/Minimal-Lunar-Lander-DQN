import gymnasium as gym
from environment.lunar_lander_env import create_lunar_lander_env
from stable_baselines3 import DQN
from agent.deep_q_network import dqn_agent
from agent.action import interpret_action
from agent.observation import interpret_observation
from train_model.train import train_model
from testing.evaluate import evaluate

# 1. Lunar Lander environment
env = create_lunar_lander_env(gym)

# 2. Deep Q-Network (DQN) agent using Stable Baselines3
model = dqn_agent(env, DQN)

# 3. Train the model
model_train = train_model(env, model)


# 4. Evaluate the trained model
evaluate(env, model_train)