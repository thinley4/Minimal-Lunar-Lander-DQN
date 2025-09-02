
# Action Space

# There are four discrete actions available:
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine

# continuous = False means the action space is discrete (4 actions)

def create_lunar_lander_env(gym):
    return gym.make("LunarLander-v3", render_mode="rgb_array", continuous=False, gravity=-10.0, enable_wind=False)


# render_mode
# rgb_array needed for video recording
# human for rendering