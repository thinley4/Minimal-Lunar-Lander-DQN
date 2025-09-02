Minimal Lunar Lander RL Project

1. Setup Environment:

Use Gymnasium's Lunar Lander environment with discrete actions (engine on/off).

Keep default parameters for gravity and wind disabled.

2. Simple Agent:

Implement a basic Deep Q-Network (DQN) agent using Stable Baselines3 or a simple neural network from scratch in PyTorch or TensorFlow.

The agent observes the state vector (position, velocity, angle, leg contact) and chooses one of four discrete actions.

3. Training Loop:

Train the agent over episodes, collecting experience and updating the DQN.

Use an epsilon-greedy policy for exploration.

Keep training steps manageable (e.g., 10,000 to 50,000 steps).

4. Evaluation:

Periodically run test episodes without exploration to track the agent's landing success and score.

Log and plot the reward progress over training episodes.

5. Basic Visualization:

Use Gymnasium’s built-in render() to visualize the lander during test runs.

Optionally save videos/gifs of successful landings to review progress.