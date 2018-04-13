"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to train and evaluate certain learning algorithm.
"""

from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from learning_interface import deep_q_learning
from network_tools import network_info as ni


""" Create an environment that run the simulation till fail for episode_num times """
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
env_name = 'CartPole-v0'
environment = ei.EnvInterface(env_name)
episode_num = 500
environment.set_up(episode_num)

""" Create the network for generating actions """
network_input_height = 40
network_input_width  = 80
network_batch_size   = [16, 32, 32]
network_kernel_stride_pad = [[5, 2, 0], [5, 2, 0], [5, 2, 0]]

liner_feature_num = ni.get_liner_input_dim(network_input_height, network_input_width,
                                           network_batch_size[-1], network_kernel_stride_pad)
shape_param = [liner_feature_num, environment.env.action_space.n]
# BATCH_SIZE
# GAMMA
# EPS_START
# EPS_END
# EPS_DECAY
# TARGET_UPDATE
hyper_param = [128, 0.999, 0.9, 0.05, 200, 250]
# Path and file name for trained model
path = ['/home/omnisky/', env_name]
network = deep_q_learning.DQNInterface(shape_param, hyper_param, path)

# Create the memory for storing training data
memory = network.memory

# Loop over until all episodes have been executed
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

    # Optimize the model using memory
    network.set_done(environment.get_progress())
    network.train()

    result = environment.get_progress()
    pp.pretty_print(action[0, 0], result)

environment.reset_env()
