"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to train and evaluate certain learning algorithm.
"""
from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from learning_interface import visual_dqn
from network_tools import network_info as ni
from visualize_interface import tensorboard_interface as ti
# from visualize_interface import vis_interface as vi

""" Create an environment that run the simulation till fail for episode_num times """
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
env_name = 'CartPole-v0'
episode_num = 10000
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
# REWARD_EXPAND
# MODEL_UPDATE
# MODEL_SAVE
hyper_param = [128, 0.99, 0.7, 0.1, 200, 0.1, 500, 500]
# Path and file name for trained model
path = ['/home/omnisky/cartpole_exp/exp8/', env_name]
network = visual_dqn.DQNTrainInterface(shape_param, hyper_param, path, use_trained=None)

"""Create the memory for storing training data"""
memory = network.memory

"""Create visualization interface"""
# visualization = vi.VisInterface("state")
tensorboard = ti.TensorboardInterface(path[0], path[1])

# Loop over until all episodes have been executed
counter = 0
while not environment.finished:
    prev_state = environment.get_last_state()

    # visualization.plot_image(prev_state.cpu().squeeze(0).permute(1, 2, 0).numpy())
    action, use_net = network.select_action(prev_state)
    environment.step_once(action[0, 0])
    counter += 1

    observation = environment.observation  # object reward done info
    # Set current finished episodes and steps for the network
    reward_tensor, reward = network.set_reward(observation[1])

    # Store the procedure into memory
    memory.push(prev_state, action, environment.curr_state, reward_tensor)

    # Optimize the model using memory
    network.train()

    pp.pretty_print([network.episode_done, network.steps_done],
                    action[0, 0], reward, use_net)

    network.update(observation[-2], float(counter)/1000)

    tensorboard.add_scalar('loss', network.loss, counter)
