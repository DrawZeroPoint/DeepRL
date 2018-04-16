"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to train and evaluate certain learning algorithm.
"""
from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from learning_interface import deep_q_learning
from network_tools import network_info as ni
from visualize_interface import vis_interface as vi

""" Create an environment that run the simulation till fail for episode_num times """
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
env_name = 'CartPole-v0'
episode_num = 1000
environment = ei.EnvInterface(env_name, episode_num)


""" Create the network for generating actions """
network_input_height = 40
network_input_width  = 80
network_batch_size   = [16, 32, 32]
network_kernel_stride_pad = [[5, 2, 0], [5, 2, 0], [5, 2, 0]]

liner_input_num = ni.get_liner_input_dim(network_input_height, network_input_width,
                                         network_batch_size[-1], network_kernel_stride_pad)
shape_param = [liner_input_num, environment.env.action_space.n]
# BATCH_SIZE
# GAMMA
# EPS_START
# EPS_END
# EPS_DECAY
# TARGET_UPDATE
hyper_param = [128, 0.999, 0.9, 0.05, 200, 500]
# Path and file name for trained model
path = ['/home/omnisky/', env_name]
network = deep_q_learning.DQNTrainInterface(shape_param, hyper_param, path)

"""Create the memory for storing training data"""
memory = network.memory

"""Create visualization interface"""
# visualization = vi.VisInterface("state")

# Loop over until all episodes have been executed
while not environment.finished:
    prev_state = environment.get_last_state()

    # visualization.plot_image(prev_state.cpu().squeeze(0).permute(1, 2, 0).numpy())
    action = network.select_action(prev_state)
    environment.step_once(action[0, 0])

    observation = environment.observation
    reward = network.tensor([observation[1]])  # object reward done info

    # Store the procedure into memory
    memory.push(prev_state, action, environment.curr_state, reward)

    # Optimize the model using memory
    network.set_done(environment.episode_count)
    network.train()

    pp.pretty_print([environment.episode_count, environment.step_count], action[0, 0], observation[1])

