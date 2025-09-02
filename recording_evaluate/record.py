import numpy as np
from agent.action import interpret_action
from agent.observation import interpret_observation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# Configuration
num_eval_episodes = 5

def record(env, model_train):

    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="video-LunarLander",    # Folder to save videos
        name_prefix="final",               # Prefix for video filenames
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    print(f"Videos will be saved to: video-LunarLander/")

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        episode_over = False
        while not episode_over:

            action, _ = model_train.predict(obs, deterministic=True)
            action_int = int(action)
            
            print(f"  Observation: {interpret_observation(obs)}")
            print(f"  Chosen action: {action_int} ({interpret_action(action_int)})")
            
            # Execute action and get new observation
            obs, reward, terminated, truncated, info = env.step(action_int)

            episode_reward += reward
            step_count += 1

            episode_over = terminated or truncated

        print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

    env.close()

    # Print summary statistics
    print(f'\nEvaluation Summary:')
    print(f'Episode durations: {list(env.time_queue)}')
    print(f'Episode rewards: {list(env.return_queue)}')
    print(f'Episode lengths: {list(env.length_queue)}')

    # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Average episode length: {avg_length:.1f} steps')
    print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')