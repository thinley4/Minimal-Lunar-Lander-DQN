from agent.observation import interpret_observation
from agent.action import interpret_action

def evaluate(env, model_train):
    # After training, use the trained policy
    print("\nAfter training - using trained policy:")
    obs, info = env.reset()
    for step in range(10):
        env.render() # For visualization, if supported
        # Agent chooses action based on current observation
        action, _ = model_train.predict(obs, deterministic=True)
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