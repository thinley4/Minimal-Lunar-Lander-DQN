# 2. The agent observes the state vector (position, velocity, angle, leg contact) and 
# chooses one of four discrete actions.

def interpret_observation(obs):
    """
    LunarLander observation space has 8 dimensions:
    [0] x position
    [1] y position  
    [2] x velocity
    [3] y velocity
    [4] angle
    [5] angular velocity
    [6] left leg contact (0 or 1)
    [7] right leg contact (0 or 1)
    """
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, angular_vel = obs[4], obs[5]
    left_leg_contact = bool(obs[6])
    right_leg_contact = bool(obs[7])
    
    return {
        'position': (x_pos, y_pos),
        'velocity': (x_vel, y_vel),
        'angle': angle,
        'angular_velocity': angular_vel,
        'leg_contact': (left_leg_contact, right_leg_contact)
    }