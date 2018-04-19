from __future__ import print_function
from termcolor import colored


def pretty_print(progress, action, reward, use_net):
    text = 'Episode %06d, step %03d | action: %d | reward: %-.1f' % (progress[0], progress[1], action, reward)
    # text = "Episode {}, step {} | action: {} | reward: {}".format(progress[0], progress[1], action, reward)
    if use_net:
        content = colored(text, 'yellow')
        print(content)
    else:
        print(text)
