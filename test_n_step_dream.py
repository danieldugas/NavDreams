import os
import numpy as np
import random
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire
from tqdm import tqdm
import copy

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navdreams.rssm import RSSMWMConf, RSSMWorldModel
from navdreams.rssm_a0 import RSSMA0WMConf, RSSMA0WorldModel
from navdreams.tssm import TSSMWMConf, TSSMWorldModel
from navdreams.transformerL import TransformerLWMConf, TransformerLWorldModel
from navdreams.worldmodel import DummyWorldModel, GreyDummyWorldModel
from navdreams.auto_debug import enable_auto_debug
from paper_sequences import load_paper_sequences
from plot_gym_training_progress import make_legend_pickable

class TransformerWorldModel(GPT):
    def fill_dream_sequence(self, real_sequence, context_length):
        """ Fills dream sequence based on context from real_sequence
            real_sequence is a list of dicts, one for each step in the sequence.
            each dict has
            "obs": numpy image (W, H, CH) [0, 1]
            "state": numpy (2,) [-inf, inf]
            "action": numpy (3,) [-inf, inf]

            context_length (int): number of steps of the real sequence to keep in the dream sequence

            output:
            dream_sequence: same length as the real_sequence, but observations and states are predicted
                    open-loop by the worldmodel, while actions are taken from the real sequence
            """
        T = self.get_block_size()
        sequence_length = len(real_sequence)
        if sequence_length > T:
            print("Warning: sequence_length > block_size ({} > {} in {})!".format(
                sequence_length, T, type(self).__name__))
        dream_sequence = copy.deepcopy(real_sequence[:context_length])
        dream_sequence[-1]['action'] = None
        real_actions = [d['action'] for d in real_sequence]
        next_actions = real_actions[context_length-1:sequence_length-1]
        for action in next_actions:
            dream_sequence[-1]['action'] = action * 1.
            img_npred, goal_pred = self.get_next(dream_sequence[-T:])
            # update sequence
            dream_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
        dream_sequence[-1]['action'] = next_actions[-1] * 1.
        return dream_sequence

def single_sequence_n_step_error(real_sequence, dream_sequence, dones, context_length):
    sequence_length = len(real_sequence)
    dream_length = sequence_length - context_length
    dream_obs = np.array([d["obs"] for d in dream_sequence[context_length:]]) # (D, W, H, C)
    dream_vecobs = np.array([d["state"] for d in dream_sequence[context_length:]])
    real_obs = np.array([d["obs"] for d in real_sequence[context_length:]])
    real_vecobs = np.array([d["state"] for d in real_sequence[context_length:]])
    obs_error = np.mean( # mean over all pixels and channels
        np.reshape(np.square(dream_obs - real_obs), (dream_length, -1)),
        axis=-1) # now (D,)
    vecobs_error = np.mean(
        np.reshape(np.square(dream_vecobs - real_vecobs), (dream_length, -1)),
        axis=-1) # now (D, )
    # if a reset is in the sequence, ignore predictions for subsequent frames
    ignore_error = np.cumsum(dones[context_length:]) > 0 # (D,)
    obs_error[ignore_error] = np.nan
    vecobs_error[ignore_error] = np.nan
    return obs_error, vecobs_error

def worldmodel_n_step_error(worldmodel, test_dataset_folder,
                            sequence_length=32, context_length=16, samples=0):
    # parameters
    shuffle = True
    dream_length = sequence_length - context_length
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=None)
    N = min(samples, len(seq_loader))
    if samples == 0:
        N = len(seq_loader)
    obs_error = np.ones((N, dream_length)) * np.nan
    vecobs_error = np.ones((N, dream_length)) * np.nan
    indices = list(range(len(seq_loader)))
    if shuffle:
        random.Random(4).shuffle(indices)
    pbar = tqdm(indices[:N])
#     for i, (x, a, y, x_rs, y_rs, dones) in enumerate(seq_loader):
    for i, idx in enumerate(pbar):
        if idx >= len(seq_loader): # this shouldn't be necessary, but it is (len is not honored by for)
            continue
        x, a, y, x_rs, y_rs, dones = seq_loader[idx]
        real_sequence = [dict(obs=x[j], state=x_rs[j], action=a[j]) for j in range(sequence_length)]
        dream_sequence = worldmodel.fill_dream_sequence(real_sequence, context_length)
        obs_error[i], vecobs_error[i] = single_sequence_n_step_error(
            real_sequence, dream_sequence, dones, context_length)
        if i % 10 == 0:
            pbar.set_description(
                f"1-step error {np.nanmean(obs_error, axis=0)[0]:.5f} \
                  16-step error {np.nanmean(obs_error, axis=0)[15]:.5f}"
            )
    archive_path = "/tmp/{}_n_step_errors.npz".format(type(worldmodel).__name__)
    np.savez_compressed(archive_path, obs_error=obs_error, vecobs_error=vecobs_error)
    print(f"Saved n-step errors to {archive_path}")
    mean_obs_error = np.nanmean(obs_error, axis=0)
    mean_vecobs_error = np.nanmean(vecobs_error, axis=0)
    return mean_obs_error, mean_vecobs_error

def hide_axes_but_keep_ylabel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if False:
        ax.set_axis_off()

def sequence_to_gif(dream_sequence, worldmodel_name, real_sequence=None, sequence_idx=0):
    from moviepy.editor import ImageSequenceClip
    dreamframes = [(d["obs"] * 255).astype(np.uint8) for d in dream_sequence]
    frames = dreamframes
    if real_sequence is not None:
        realframes = [(d["obs"] * 255).astype(np.uint8) for d in real_sequence]
        frames = [np.concatenate([r, d], axis=0) for r, d in zip(realframes, dreamframes)]
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_gif("/tmp/dream_of_length{}_index{}_{}.gif".format(
        len(frames), sequence_idx, worldmodel_name), fps=20)

def sequences_to_comparison_gif(dream_sequences, worldmodel_names, real_sequence, sequence_idx=0):
    from moviepy.editor import ImageSequenceClip
    all_dreamframes = [[(d["obs"] * 255).astype(np.uint8) for d in dream_sequence]
                       for dream_sequence in dream_sequences]
    realframes = [(d["obs"] * 255).astype(np.uint8) for d in real_sequence]
    frames = [np.concatenate(imglist, axis=0) for imglist in zip(realframes, *all_dreamframes)]
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_gif("/tmp/comparison_dream_of_length{}_index{}_{}.gif".format(
        len(frames), sequence_idx, '_'.join([n[:6] for n in worldmodel_names])), fps=20)

def black_sequence_after_done(dream_sequence):
    done = False
    for dic in dream_sequence:
        img = dic["obs"]
        if np.mean(img) < 0.1:
            done = True
        if done:
            dic["obs"] = np.zeros_like(img)

def main(dataset="SCR",
         gpu=False,
         dream_length=48,
         context_length=16,
         n_examples=2,
         error=False,
         dataset_info=False,
         offset=0,
         skip=2,
         samples=1000,
         gifs=False,
         paper_sequences=False,
         ):
    sequence_length = dream_length + context_length
    worldmodel_types = ["TransformerL_V0", "RSSM_A1", "RSSM_A0", "TSSM_V2", "transformer"]
    worldmodel_types = ["transformer", "RSSM_A0_explicit", "TransformerL_V0", "RSSM_A0", "TSSM_V2"]
#     worldmodel_types = ["RSSM_A0_explicit", "RSSM_A0"]
    worldmodel_types = ["TransformerL_V0", "RSSM_A0", "transformer"] # comparison image
#     worldmodel_types = ["TransformerL_V0", "transformer", "TSSM_V2", "RSSM_A0"] # gifs
#     worldmodel_types = ["DummyWorldModel"]
    discrete_actions = False

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navdreams_data/wm_experiments/datasets/V/rosbag")]
        examples = [34, 51, 23, 42, 79, 5, 120]
        examples = [0, 1500, 3000, 4500, 6000, 1000, 4000] # for length 64
        examples = [0, 3000, 6000, 9000, 12000, 1000, 4000]
        examples = [3004, 4002] # for ctx 16 drm 48
        examples = [
            3, 35,
            61, # city 39
            50, 53, 70, 87, 91, 92, # asl 40
            51, # alt
        ]
        examples = np.random.permutation(examples)
        examples = [3, 92]
        print("examples:", examples)
    elif dataset == "staticasl":
        dataset_dir = [os.path.expanduser("~/navdreams_data/wm_experiments/datasets/V/navrep3dasl")]
        examples = [34, 51, 23, 42, 79, 5, 120]
    elif dataset == "dSalt":
        dataset_dir = [os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/discrete_navrep3dalt")]
        examples = [34, 51, 23, 42, 79, 5, 120]
        discrete_actions = True
        worldmodel_types = ["TransformerL_V0"]
    elif dataset == "qSalt":
        dataset = "Salt"
        dataset_dir = [os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/quantized_navrep3dalt")]
        examples = [34, 51, 23, 42, 79, 5, 120]
        worldmodel_types = ["TransformerL_V0", "transformer"]
    else:
        raise NotImplementedError(dataset)
    examples = [idx + offset for idx in examples]

    if error:
        worldmodel_types = worldmodel_types + ["DummyWorldModel", "GreyDummyWorldModel"]

    def load_worldmodels(worldmodel_types):
        print("Loading worldmodels...")
        worldmodels = []
        for worldmodel_type in worldmodel_types:
            if worldmodel_type == "transformer":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/transformer_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                BLOCK_SIZE = 32
                _H = 64
                _C = 3
                mconf = GPTConfig(BLOCK_SIZE, _H)
                mconf.image_channels = _C
                if discrete_actions:
                    mconf.n_action = 4
                model = TransformerWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "RSSM_A1":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/RSSM_A1_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = RSSMWMConf()
                mconf.image_channels = 3
                model = RSSMWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "RSSM_A0" or worldmodel_type == "RSSM_A0_explicit":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/RSSM_A0_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = RSSMA0WMConf()
                mconf.image_channels = 3
                model = RSSMA0WorldModel(mconf, gpu=gpu)
                if worldmodel_type == "RSSM_A0_explicit":
                    class RSSMA0WorldModelExplicit(RSSMA0WorldModel):
                        noop = 0
                    model = RSSMA0WorldModelExplicit(mconf, gpu=gpu)
                    model.fill_dream_sequence = model.fill_dream_sequence_through_images
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "TSSM_V2":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/TSSM_V2_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = TSSMWMConf()
                mconf.image_channels = 3
                model = TSSMWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "TransformerL_V0":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/TransformerL_V0_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = TransformerLWMConf()
                mconf.image_channels = 3
                if discrete_actions:
                    mconf.n_action = 4
                model = TransformerLWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "DummyWorldModel":
                worldmodel = DummyWorldModel(gpu=gpu)
            elif worldmodel_type == "GreyDummyWorldModel":
                worldmodel = GreyDummyWorldModel(gpu=gpu)
            else:
                raise NotImplementedError
            worldmodels.append(worldmodel)
        print("Done.")
        return worldmodels

    worldmodels = load_worldmodels(worldmodel_types)

    if gifs:
        # parameters
        shuffle = True
        # load dataset
        seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                       channel_first=False, as_torch_tensors=False, file_limit=None)
        N = min(samples, len(seq_loader))
        if samples == 0:
            N = len(seq_loader)
        obs_error = np.ones((N, dream_length)) * np.nan
        vecobs_error = np.ones((N, dream_length)) * np.nan
        indices = list(range(len(seq_loader)))
        if shuffle:
            random.Random(4).shuffle(indices)
        pbar = tqdm(indices[:N])
        for i, idx in enumerate(pbar):
            if idx >= len(seq_loader): # this shouldn't be necessary, but it is (len is not honored by for)
                continue
            x, a, y, x_rs, y_rs, dones = seq_loader[idx]
            if np.any(dones[:-1]):
                continue
            real_sequence = [dict(obs=x[j], state=x_rs[j], action=a[j]) for j in range(sequence_length)]
            dream_sequences = []
            names = []
            for worldmodel in worldmodels:
                black_sequence_after_done(real_sequence)
                dream_sequence = worldmodel.fill_dream_sequence(real_sequence, context_length)
                black_sequence_after_done(dream_sequence)
                dream_sequences.append(dream_sequence)
                names.append(type(worldmodel).__name__)
                sequence_to_gif(dream_sequence, type(worldmodel).__name__, real_sequence, idx)
            sequences_to_comparison_gif(dream_sequences, names, real_sequence, idx)
        return

    if error:
        print("Computing n-step error")
        n_step_errors = []
        for worldmodel in worldmodels:
            obs_n_step_error, vecobs_n_step_error = worldmodel_n_step_error(
                worldmodel, dataset_dir, sequence_length=sequence_length,
                context_length=context_length, samples=samples)
            n_step_errors.append((obs_n_step_error, vecobs_n_step_error))
        fig, (ax1, ax2) = plt.subplots(1, 2, num="n-step error")
        linegroups = []
        legends = worldmodel_types
        for obs_n_step_error, vecobs_n_step_error in n_step_errors:
            line1, = ax1.plot(obs_n_step_error)
            line2, = ax2.plot(vecobs_n_step_error)
            linegroups.append([line1, line2])
        L = fig.legend([lines[0] for lines in linegroups], legends)
        make_legend_pickable(L, linegroups)
        plt.savefig("/tmp/n_step_errors.png")
        print("Saved figure.")
#         plt.show()
        return

    if paper_sequences:
        example_sequences = load_paper_sequences(examples, n_examples, dataset_dir, sequence_length)
    else:
        # load example sequences
        seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                       channel_first=False, as_torch_tensors=False, file_limit=None)

        print("{} sequences available".format(len(seq_loader)))
        example_sequences = {examples[i]: None for i in range(n_examples)}
        for idx in example_sequences:
            if idx >= len(seq_loader):
                raise IndexError("{} is out of range".format(idx))
            (x, a, y, x_rs, y_rs, dones) = seq_loader[idx]
            example_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i], done=dones[i])
                                for i in range(sequence_length)]
            example_sequences[idx] = example_sequence

    # used to better understand the dataset: average length of a sequence (until done),
    # average error of a grey image
    if dataset_info:
        # average done
        true_lengths = []
        for x, a, y, x_rs, y_rs, dones in tqdm(seq_loader):
            if len(dones) == 0:
                continue
            true_length = np.argmax(dones > 0)
            if true_length == 0:
                true_length = len(dones)
            if dones[0]:
                true_length = 0
            true_lengths.append(true_length)
        median_true_length = np.nanmedian(true_lengths)
        proportion_undended = np.sum(true_lengths == np.max(true_lengths)) / len(true_lengths)
        print("Median true length of a sequence: {}".format(median_true_length))
        print("percent unended: {:.1f}%".format(100. * proportion_undended))
        _, (ax1, ax2) = plt.subplots(2, 1, num="histogram")
        ax1.hist(true_lengths, bins=sequence_length)
        ax2.hist(true_lengths, bins=sequence_length, cumulative=-1)
        # grey error
        n_step_errors = []
        for worldmodel in [GreyDummyWorldModel(gpu=False)]:
            obs_n_step_error, vecobs_n_step_error = worldmodel_n_step_error(
                worldmodel, dataset_dir, sequence_length=sequence_length,
                context_length=context_length, samples=samples)
            n_step_errors.append((obs_n_step_error, vecobs_n_step_error))
        fig, (ax1, ax2) = plt.subplots(1, 2, num="n-step error")
        x = np.arange(context_length, sequence_length)
        for obs_n_step_error, vecobs_n_step_error in n_step_errors:
            line1, = ax1.plot(x, obs_n_step_error)
            line2, = ax2.plot(x, vecobs_n_step_error)
        plt.axvline(x=median_true_length, color="r")
        plt.show()
        raise ValueError("No error: raising to allow inspection")
        return

    # fill dream sequences from world model
    example_filled_sequences = []
    for n, idx in enumerate(tqdm(example_sequences)):
        if example_sequences[idx] is None:
            continue
        real_sequence = example_sequences[idx]
        dones = [step["done"] for step in real_sequence]
        dream_sequences = []
        for worldmodel in worldmodels:
            dream_sequences.append(worldmodel.fill_dream_sequence(real_sequence, context_length))
        example_filled_sequences.append((real_sequence, dream_sequences, dones))

    name = "-".join([str(n) for n in example_sequences])

    # error plot
    fig2, axes2 = plt.subplots(n_examples, 2, num="n-step err")
    axes2 = np.array(axes2).reshape((-1, 2))
    for n, (real_sequence, dream_sequences, dones) in enumerate(example_filled_sequences):
        linegroups = []
        legends = worldmodel_types[:]
        for dream_sequence in dream_sequences:
            obs_error, vecobs_error = single_sequence_n_step_error(
                real_sequence, dream_sequence, dones, context_length)
            line1, = axes2[n, 0].plot(obs_error)
            line2, = axes2[n, 1].plot(vecobs_error)
            linegroups.append([line1, line2])
        L = fig2.legend([lines[0] for lines in linegroups], legends)
        make_legend_pickable(L, linegroups)
    fig2.savefig("/tmp/n_step_error_for_dream_comparison_{}.png".format(name))

    # images plot
    n_rows_per_example = (len(worldmodels) + 1)
    n_cols = sequence_length // (1 + skip)
    if skip == 3: # dirty but no time
        n_cols += 1
    fig, axes = plt.subplots(n_rows_per_example * n_examples, n_cols, num="dream",
                             figsize=(22, 3 * n_examples), dpi=100)
    axes = np.array(axes).reshape((-1, n_cols))
    for n, (real_sequence, dream_sequences, dones) in enumerate(example_filled_sequences):
        for i in range(sequence_length):
            axes[n_rows_per_example*n, i // (1 + skip)].imshow(real_sequence[i]['obs'])
            if n == 0:
                axes[n_rows_per_example*n, i // (1 + skip)].set_title(i, fontsize=15)
            if i >= context_length:
                for m, dream_sequence in enumerate(dream_sequences):
                    axes[n_rows_per_example*n+1+m, i // (1 + skip)].imshow(dream_sequence[i]['obs'])
        axes[n_rows_per_example*n, -1].set_ylabel("GT", rotation=0, labelpad=50)
        axes[n_rows_per_example*n, -1].yaxis.set_label_position("right")
        for m in range(len(dream_sequences)):
            axes[n_rows_per_example*n+1+m, -1].set_ylabel("{}".format(worldmodel_types[m]),
                                                          rotation=0, labelpad=50)
            axes[n_rows_per_example*n+1+m, -1].yaxis.set_label_position("right")
        for ax in np.array(axes).flatten():
            hide_axes_but_keep_ylabel(ax)
            plt.subplots_adjust(wspace=0.1)
    fig.savefig("/tmp/dream_comparison_{}.png".format(name), dpi=100)

    print("Saved figures.")
    plt.show()


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
