from agent.observation import interpret_observation
from agent.action import interpret_action

def train_model(env, model):
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

    return model.learn(total_timesteps=10000)