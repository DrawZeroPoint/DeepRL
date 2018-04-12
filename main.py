"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to evaluate the performance of certain learning algorithm. Using this template we only need to give the
action to be taken using customized method.
"""

from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from learning_tools import deep_q_learning


# Create an environment that run the simulation till fail for episode_num times
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
environment = ei.EnvInterface('CartPole-v0')
episode_num = 500
environment.set_up(episode_num)

# Create the network for generating actions
# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10
param = [128, 0.999, 0.9, 0.05, 200, 250]
network = deep_q_learning.DQNInterface(param)

# Create the memory for storing training data
memory = network.memory

while not environment.get_progress()[2]:
    # Get the state before taking action
    prev_state = environment.get_state()

    action = network.select_action(prev_state)
    environment.step_once(action[0, 0])

    observation = environment.observation
    reward = network.tensor([observation[1]])  # object reward done info

    # Get the state after taking action
    curr_state = environment.get_state()

    # Store the procedure into memory
    memory.push(prev_state, action, curr_state, reward)

    # Optimize the model
    network.set_done(environment.get_progress())
    network.train()

    result = environment.get_progress()
    pp.pretty_print(action[0, 0], result)

environment.reset_env()
