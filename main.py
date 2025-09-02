from environment.lunar_lander_env import create_lunar_lander_env
from agent.deep_q_network import dqn_agent
from agent.action import interpret_action
from agent.observation import interpret_observation


# 1. Lunar Lander environment
env = create_lunar_lander_env()

# 2. Deep Q-Network (DQN) agent using Stable Baselines3
model = dqn_agent()




# Reset environment to start a new episode
observation, info = env.reset()

# Example of how the agent observes and acts
print("Initial observation:")
state_info = interpret_observation(observation)
print(f"Position: {state_info['position']}")
print(f"Velocity: {state_info['velocity']}")
print(f"Angle: {state_info['angle']:.3f}")
print(f"Leg contact: {state_info['leg_contact']}")

# Before training, actions are random
print("\nBefore training - random action:")
random_action = env.action_space.sample()
print(f"Action {random_action}: {interpret_action(random_action)}")

# Take the random action
next_obs, reward, terminated, truncated, info = env.step(random_action)
print(f"Reward received: {reward}")

# Train the model
print("\nTraining the DQN agent...")
model.learn(total_timesteps=10000)

# After training, use the trained policy
print("\nAfter training - using trained policy:")
obs, info = env.reset()
for step in range(10):
    env.render() # For visualization, if supported
    # Agent chooses action based on current observation
    action, _ = model.predict(obs, deterministic=True)
    action_int = int(action)
    
    print(f"Step {step + 1}:")
    print(f"  Observation: {interpret_observation(obs)}")
    print(f"  Chosen action: {action_int} ({interpret_action(action_int)})")
    
    # Execute action and get new observation
    obs, reward, terminated, truncated, info = env.step(action_int)
    print(f"  Reward: {reward:.3f}")
    
    if terminated or truncated:
        print("  Episode ended!")
        obs, info = env.reset()
        break

env.close()