# import torch
# from gym_interface import gym_info as gi
# from network_tools import network_info as ni
#
# # gi.list_envs()
# # print(ni.get_blob_shape(40, 60, [[5, 2, 0], [5, 2, 0], [5, 2, 0]]))
# # print(ni.get_liner_input_dim(40, 80, 32, [[5, 2, 0], [5, 2, 0], [5, 2, 0]]))
#
# a = torch.randn(4, 4)
# y = a.view(16)
# print(y)
# b = y.max(-1)
# print(b)

# import gym
#
# env = gym.make('LunarLander-v2')
#
# print env.observation_space
# print env.action_space
#
# for i_episode in range(100):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break

from visualize_interface import vis_interface as vi
import numpy as np

vis = vi.VisInterface('example')
x = np.diag(np.arange(2, 12))[::-1]
vis.plot_image(x)
pass
