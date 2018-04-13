from gym_interface import gym_info as gi
from network_tools import network_info as ni

gi.list_envs()
# print(ni.get_blob_shape(40, 60, [[5, 2, 0], [5, 2, 0], [5, 2, 0]]))
print(ni.get_liner_input_dim(40, 80, 32, [[5, 2, 0], [5, 2, 0], [5, 2, 0]]))
