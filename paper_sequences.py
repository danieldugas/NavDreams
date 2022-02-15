import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire
from tqdm import tqdm
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep3d.auto_debug import enable_auto_debug

def find_files(dir_):
    files = []
    dir_ = os.path.expanduser(dir_)
    for f in os.listdir(dir_):
        if "_sequence_" in f and f.endswith(".pkl"):
            files.append(os.path.join(dir_, f))
    return files

def generate_paper_sequences(dir_, dataset_dir, sequence_length):
    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=None)
    print("{} sequences available".format(len(seq_loader)))
    print("Saving sequences")
    i = 0
    for idx in tqdm(np.random.permutation(range(len(seq_loader)))):
        (x, a, y, x_rs, y_rs, dones) = seq_loader[idx]
        if np.any(dones):
            continue
        real_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i], done=dones[i])
                         for i in range(sequence_length)]
        path = os.path.join(dir_, "{}_sequence_{}.pkl".format(sequence_length, i))
        pickle.dump(real_sequence, open(path, "wb"))
        _ = pickle.load(open(path, "rb"))
        i += 1

def load_paper_sequences(examples, n_examples, dataset_dir, sequence_length,
                         dir_="~/navrep3d_test/sequences/"):
    dir_ = os.path.expanduser(dir_)
    make_dir_if_not_exists(dir_)
    # list all data files
    files = find_files(dir_)
    if len(files) < n_examples:
        print("No sequences found. Generate?")
        if not input("y/n: ").lower().startswith("y"):
            raise ValueError("Debug: Program end")
            return
        generate_paper_sequences(dir_, dataset_dir, sequence_length)
    files = find_files(dir_)
    file_dict = {}
    for file in files:
        idx = int(file.split("_")[-1].split(".")[0])
        file_dict[idx] = file
    example_sequences = {examples[i]: None for i in range(n_examples)}
    for idx in example_sequences:
        if idx not in file_dict:
            raise ValueError("No sequence found for example {}".format(idx))
        file = file_dict[idx]
        loaded_sequence = pickle.load(open(file, "rb"))
        example_sequences[idx] = loaded_sequence
    return example_sequences

def hide_axes_but_keep_ylabel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if False:
        ax.set_axis_off()

def plot_sequence(sequence, skip=1, title=""):
    sequence_length = len(sequence)
    # images plot
    n_rows = 1
    n_cols = sequence_length // (1 + skip)
    fig, axes = plt.subplots(n_rows, n_cols, num="dream",
                             figsize=(22, 14), dpi=100)
    axes = np.array(axes).reshape((-1, n_cols))
    n = 0
    for i in range(sequence_length):
        axes[n_rows*n, i // (1 + skip)].imshow(sequence[i]['obs'])
    axes[n_rows*n, -1].set_ylabel("GT", rotation=0, labelpad=50)
    axes[n_rows*n, -1].yaxis.set_label_position("right")
    for ax in np.array(axes).flatten():
        hide_axes_but_keep_ylabel(ax)
        plt.subplots_adjust(wspace=0.1)
    plt.title(title)
    plt.show()

def main():
    N = 100
    dataset_dir = [os.path.expanduser("~/navrep3d_test/datasets/V/navrep3dalt"),
                   os.path.expanduser("~/navrep3d_test/datasets/V/navrep3dcity"),
                   os.path.expanduser("~/navrep3d_test/datasets/V/navrep3doffice"),
                   os.path.expanduser("~/navrep3d_test/datasets/V/navrep3dasl"),
                   os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
    example_sequences = load_paper_sequences(range(N), N, dataset_dir, 64)
    for idx in example_sequences:
        sequence = example_sequences[idx]
        plot_sequence(sequence, title=str(idx) + "_sequence_64.pkl")


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
