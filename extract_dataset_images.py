import os
from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire
from tqdm import tqdm

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navdreams.rssm import RSSMWMConf, RSSMWorldModel
from navdreams.rssm_a0 import RSSMA0WMConf, RSSMA0WorldModel
from navdreams.tssm import TSSMWMConf, TSSMWorldModel
from navdreams.transformerL import TransformerLWMConf, TransformerLWorldModel
from navdreams.worldmodel import fill_dream_sequence, DummyWorldModel, GreyDummyWorldModel
from navdreams.auto_debug import enable_auto_debug
from plot_gym_training_progress import make_legend_pickable

def main():
    dataset_dir = [
#         os.path.expanduser("~/navrep3d_test/datasets/V/navrep3dalt"),
#         os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
#         os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
#         os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
#         os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dkozehd"),
        os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dgallery"),
        os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcathedral"),
#         os.path.expanduser("~/navrep3d_W/datasets/V/rosbag"),
    ]
    sequence_length = 8

    # parameters
    shuffle = True
    # load dataset
    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=None)
    N = len(seq_loader)
    indices = list(range(len(seq_loader)))
    if shuffle:
        random.Random(5).shuffle(indices)
    pbar = tqdm(indices[:N])
    n = 0
    for i, idx in enumerate(pbar):
        if idx >= len(seq_loader): # this shouldn't be necessary, but it is (len is not honored by for)
            continue
        x, a, y, x_rs, y_rs, dones = seq_loader[idx]
        for image in x:
            plt.imsave("/tmp/image_{:05}.png".format(n), image)
            n += 1
            if n >= 1000:
                break


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
