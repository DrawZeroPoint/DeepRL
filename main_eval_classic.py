"""
This demo illustrated how to establish an environment with OpenAI Gym and iterate for several episodes
to evaluate certain learning algorithm.
"""

from gym_interface import env_interface as ei
from io_tools import pretty_print as pp
from io_tools import file_interface as fi
from learning_interface import classic_dqn
from network_tools import network_info as ni
from visualize_interface import tensorboard_interface as ti
from visualize_interface import vis_interface as vi


""" Create an environment that run the simulation till fail for episode_num times """
# We can also set the step_num (default:0 for no limit) for each episode in set_up for limiting that
env_name = 'CartPole-v0'
episode_num = 100  # Considered solved when the average reward is greater than or equal to 195.0 over 100 trials.
environment = ei.EnvInterface(env_name, episode_num)

""" Create the network for generating actions """
shape_param = [4, 2, 164]

# Path and file name of the trained model
model_path = '/home/omnisky/cartpole_exp/cls_render/'
name_list = fi.get_file_list(model_path, env_name)

"""Create visualization interface"""
visualization = vi.VisInterface("state")
tensorboard = ti.TensorboardInterface(model_path, env_name)

iter_counter = 0.0
for file_name in name_list:

    iter_counter = file_name
    print(iter_counter)

    full_path = model_path + env_name + str(file_name)
    network = classic_dqn.NetworkEvalInterface(shape_param, full_path)

    """Create statics values"""
    episode_temp = 0
    step_count = 0
    step_temp = 0

    # Loop over until all episodes have been executed
    while not environment.finished:
        # Get the state before taking action
        state = environment.get_classic_state()

        # visualization.plot_image(prev_state.cpu().squeeze(0).permute(1, 2, 0).numpy())
        action = network.generate_action(state)
        environment.step_once_classic(action)

        observation = environment.observation
        pp.pretty_print([environment.episode_count, environment.step_count], action, observation[1], True)

        if environment.episode_count != episode_temp:
            step_count += step_temp
            episode_temp = environment.episode_count
        else:
            step_temp = environment.step_count

    environment.reset()

    average_reward = float(step_count) / episode_num

    tensorboard.add_scalar('reward', average_reward, iter_counter*1000)

    print("Problem solved: {}, average reward: {}".format(average_reward >= 195.0, average_reward))
