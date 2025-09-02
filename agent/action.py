

def interpret_action(action):
    """
    LunarLander has 4 discrete actions:
    0: Do nothing
    1: Fire left orientation engine
    2: Fire main engine
    3: Fire right orientation engine
    """
    action_meanings = {
        0: "Do nothing",
        1: "Fire left orientation engine", 
        2: "Fire main engine",
        3: "Fire right orientation engine"
    }
    return action_meanings.get(action, "Unknown action")