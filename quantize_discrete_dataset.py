import numpy as np
import os
from tqdm import tqdm
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navdreams.navrep3dtrainenv import convert_discrete_to_continuous_action

def quantize_discrete_dataset(directory):
    # list all data files
    files = []
    if isinstance(directory, list):
        raise NotImplementedError
    elif isinstance(directory, str):
        directories = [directory]
    else:
        raise NotImplementedError
    for dir_ in directories:
        dir_ = os.path.expanduser(dir_)
        for dirpath, dirnames, filenames in os.walk(dir_):
            for filename in [
                f
                for f in filenames
                if f.endswith("scans_robotstates_actions_rewards_dones.npz")
            ]:
                files.append(os.path.join(dirpath, filename))
    files = sorted(files)
    quantized_data = {}
    for path in tqdm(files):
        assert "/V/discrete_" in path
        arrays_dict = np.load(path)
        for k in arrays_dict:
            values = arrays_dict[k]
            if k == "actions":
                B, A = values.shape
                assert A == 4
                quantized_actions = []
                for action in values:
                    discrete_action = np.argmax(action)
                    quantized_actions.append(convert_discrete_to_continuous_action(discrete_action))
                values = np.array(quantized_actions)
            quantized_data[k] = values
        newpath = path.replace("/V/discrete_", "/V/quantized_")
        make_dir_if_not_exists(os.path.dirname(newpath))
        np.savez(newpath, **quantized_data)
        print("{} saved.".format(newpath))


if __name__ == "__main__":
    quantize_discrete_dataset(
        os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/discrete_navrep3dalt")
    )
