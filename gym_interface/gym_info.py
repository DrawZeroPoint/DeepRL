from gym import envs
from io_tools import pretty_print


def get_envs_name():
    env_ids = [spec.id for spec in envs.registry.all()]
    for en_vid in sorted(env_ids):
        print(en_vid)


def get_observation_value(observation, character_str):
    if character_str == 'object':
        return observation[0]
    elif character_str == 'reward':
        return observation[1]
    elif character_str == 'done':
        return observation[2]
    elif character_str == 'info':
        return observation[3]
    else:
        pretty_print.pretty_print('The second param should be \'object\', \'reward\', \'done\', or \'info\'.')
        return
