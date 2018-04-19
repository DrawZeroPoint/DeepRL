import glob


# The trained model's name follows the pattern: file_path +
# env_name + iteration_num (divided by 1000), such that we get the list
# of the iteration_num of each model in float format
def get_file_list(path, name):
    f_list = glob.glob(path + name + "*")
    name_list = []
    for f in f_list:
        prefix_len = len(path) + len(name)
        n = f[prefix_len:]
        name_list.append(float(n))
    return sorted(name_list)
