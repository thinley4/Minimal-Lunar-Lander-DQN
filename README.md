# Minimal Lunar Lander DQN RL Project

A Deep Q-Network (DQN) implementation for solving the Gymnasium Lunar Lander environment using reinforcement learning.



https://github.com/user-attachments/assets/f8244670-24e4-4fd3-892f-26e2c9cd5d2c



## Project Structure

```
lunar-lander/
├── main.py                    # Main execution script
├── environment/
│   └── lunar_lander_env.py   # Environment setup
├── agent/
│   ├── deep_q_network.py     # DQN agent configuration
│   ├── action.py             # Action interpretation
│   └── observation.py        # Observation interpretation
├── train_model/
│   └── train.py              # Training loop
└── testing/
    └── evaluate.py           # Evaluation and testing
```

## Implementation Details

### 1. Environment Setup
- Uses Gymnasium's LunarLander-v3 environment with discrete actions
- Default parameters maintained (gravity and wind settings)
- 8-dimensional observation space: position (x,y), velocity (x,y), angle, angular velocity, and leg contact sensors

### 2. Agent Implementation
- Deep Q-Network (DQN) agent using Stable Baselines3
- MLP (Multi-Layer Perceptron) policy network
- Four discrete actions available:
  - 0: Do nothing
  - 1: Fire left orientation engine
  - 2: Fire main engine  
  - 3: Fire right orientation engine

### 3. Training
- Training configured for 10,000 timesteps
- Uses DQN's built-in epsilon-greedy exploration strategy
- Experience replay and target network updates handled automatically by Stable Baselines3

### 4. Evaluation
- Post-training evaluation with deterministic policy (no exploration)
- 10-step test episodes to demonstrate learned behavior
- Detailed logging of observations, actions, and rewards
- Episode termination handling (crashed or landed)

### 5. Reward System
The environment provides rewards based on:
- Distance to landing pad (closer = higher reward)
- Landing velocity (slower = higher reward)  
- Lander orientation (horizontal = higher reward)
- Ground contact (+10 points per leg touching ground)
- Engine usage penalties (-0.03 for side engines, -0.3 for main engine)
- Landing outcome (+100 for safe landing, -100 for crash)

An episode scoring ≥200 points is considered solved.

## Usage

Run the complete training and evaluation pipeline:

```bash
python main.py
```

This will:
1. Create the Lunar Lander environment
2. Initialize the DQN agent
3. Train for 10,000 timesteps
4. Evaluate the trained model with detailed output

## Dependencies

- gymnasium
- stable-baselines3
- numpy (included with stable-baselines3)

## Output

The program displays:
- Training progress
- Post-training evaluation with step-by-step details:
  - Parsed observations (position, velocity, angle, leg contact)
  - Selected actions with descriptions
  - Received rewards
