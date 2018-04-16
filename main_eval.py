"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to evaluate certain learning algorithm.
"""

from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from learning_interface import deep_q_learning
from network_tools import network_info as ni
from visualize_interface import vis_interface as vi


""" Create an environment that run the simulation till fail for episode_num times """
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
env_name = 'CartPole-v0'
episode_num = 10
environment = ei.EnvInterface(env_name, episode_num)

""" Create the network for generating actions """
network_input_height = 40
network_input_width  = 80
network_batch_size   = [16, 32, 32]
network_kernel_stride_pad = [[5, 2, 0], [5, 2, 0], [5, 2, 0]]

liner_feature_num = ni.get_liner_input_dim(network_input_height, network_input_width,
                                           network_batch_size[-1], network_kernel_stride_pad)
shape_param = [liner_feature_num, environment.env.action_space.n]

# Path and file name of the trained model
file_name = '/home/omnisky/' + env_name + '1000'
network = deep_q_learning.DQNEvalInterface(shape_param, file_name)

"""Create visualization interface"""
visualization = vi.VisInterface("state")

# Loop over until all episodes have been executed
while not environment.finished:
    # Get the state before taking action
    prev_state = environment.get_last_state()

    # visualization.plot_image(prev_state.cpu().squeeze(0).permute(1, 2, 0).numpy())
    action = network.generate_action(prev_state)
    environment.step_once(action[0, 0])

    observation = environment.observation
    reward = network.tensor([observation[1]])  # object reward done info

    pp.pretty_print([environment.episode_count, environment.step_count], action[0, 0], observation[1])

