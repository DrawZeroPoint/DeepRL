from __future__ import print_function


def pretty_print(progress, action, reward):
    print("Episode {}, step {} | action: {} | reward: {}".format(progress[0], progress[1], action, reward))
