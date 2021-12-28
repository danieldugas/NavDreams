import os
import numpy as np
import random
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire
from tqdm import tqdm

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navrep3d.rssm import RSSMWMConf, RSSMWorldModel
from navrep3d.tssm import TSSMWMConf, TSSMWorldModel
from navrep3d.transformerL import TransformerLWMConf, TransformerLWorldModel
from navrep3d.worldmodel import fill_dream_sequence
from navrep3d.auto_debug import enable_auto_debug
from plot_gym_training_progress import make_legend_pickable

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
    assert sequence_length <= worldmodel.get_block_size()
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
        dream_sequence = fill_dream_sequence(worldmodel, real_sequence, context_length)
        obs_error[i], vecobs_error[i] = single_sequence_n_step_error(
            real_sequence, dream_sequence, dones, context_length)
        if i % 10 == 0:
            pbar.set_description(
                f"1-step error {np.nanmean(obs_error, axis=0)[0]:.5f} \
                  16-step error {np.nanmean(obs_error, axis=0)[15]:.5f}"
            )
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

def main(dataset="SCR",
         gpu=False,
         dream_length=16,
         context_length=16,
         n_examples=5,
         error=False,
         offset=0,
         samples=1000,
         ):
    sequence_length = dream_length + context_length

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
        examples = [34, 51, 23, 42, 79, 5, 120]
        examples = [0, 3000, 6000, 9000, 12000, 1000, 4000]
    elif dataset == "staticasl":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl")]
        examples = [34, 51, 23, 42, 79, 5, 120]
    else:
        raise NotImplementedError(dataset)
    examples = [idx + offset for idx in examples]

    worldmodel_types = ["transformer", "RSSM_A1", "TSSM_V2", "TransformerL_V0"]
    worldmodels = []
    for worldmodel_type in worldmodel_types:
        if worldmodel_type == "transformer":
            wm_model_path = "~/navrep3d_W/models/W/transformer_{}".format(dataset)
            wm_model_path = os.path.expanduser(wm_model_path)
            BLOCK_SIZE = 32
            _H = 64
            _C = 3
            mconf = GPTConfig(BLOCK_SIZE, _H)
            mconf.image_channels = _C
            model = GPT(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            worldmodel = model
        elif worldmodel_type == "RSSM_A1":
            wm_model_path = "~/navrep3d_W/models/W/RSSM_A1_{}".format(dataset)
            wm_model_path = os.path.expanduser(wm_model_path)
            mconf = RSSMWMConf()
            mconf.image_channels = 3
            model = RSSMWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            worldmodel = model
        elif worldmodel_type == "TSSM_V2":
            wm_model_path = "~/navrep3d_W/models/W/TSSM_V2_{}".format(dataset)
            wm_model_path = os.path.expanduser(wm_model_path)
            mconf = TSSMWMConf()
            mconf.image_channels = 3
            model = TSSMWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            worldmodel = model
        elif worldmodel_type == "TransformerL_V0":
            wm_model_path = "~/navrep3d_W/models/W/TransformerL_V0_{}".format(dataset)
            wm_model_path = os.path.expanduser(wm_model_path)
            mconf = TransformerLWMConf()
            mconf.image_channels = 3
            model = TransformerLWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            worldmodel = model
        worldmodels.append(worldmodel)

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

    example_sequences = {examples[i]: None for i in range(n_examples)}
    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=None)
    print("{} sequences available".format(len(seq_loader)))
    for idx in example_sequences:
        (x, a, y, x_rs, y_rs, dones) = seq_loader[idx]
        example_sequences[idx] = (x, a, y, x_rs, y_rs, dones)

    n_rows_per_example = (len(worldmodels) + 1)
    fig, axes = plt.subplots(n_rows_per_example * n_examples, sequence_length, num="dream",
                             figsize=(22, 14), dpi=100)
    fig2, axes2 = plt.subplots(n_examples, 2, num="n-step err")
    axes = np.array(axes).reshape((-1, sequence_length))
    axes2 = np.array(axes2).reshape((-1, 2))
    for n, idx in enumerate(tqdm(example_sequences)):
        if example_sequences[idx] is None:
            continue
        x, a, y, x_rs, y_rs, dones = example_sequences[idx]
        real_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i]) for i in range(sequence_length)]
        dream_sequences = []
        for worldmodel in worldmodels:
            dream_sequences.append(fill_dream_sequence(worldmodel, real_sequence, context_length))

        # plotting
        for i in range(sequence_length):
            axes[n_rows_per_example*n, i].imshow(real_sequence[i]['obs'])
            if i > context_length:
                for m, dream_sequence in enumerate(dream_sequences):
                    axes[n_rows_per_example*n+1+m, i].imshow(dream_sequence[i]['obs'])
        axes[n_rows_per_example*n, -1].set_ylabel("GT", rotation=0, labelpad=50)
        axes[n_rows_per_example*n, -1].yaxis.set_label_position("right")
        for m in range(len(dream_sequences)):
            axes[n_rows_per_example*n+1+m, -1].set_ylabel("{}".format(worldmodel_types[m]),
                                                          rotation=0, labelpad=50)
            axes[n_rows_per_example*n+1+m, -1].yaxis.set_label_position("right")
        for ax in np.array(axes).flatten():
            hide_axes_but_keep_ylabel(ax)
        linegroups = []
        legends = worldmodel_types
        for dream_sequence in dream_sequences:
            obs_error, vecobs_error = single_sequence_n_step_error(
                real_sequence, dream_sequence, dones, context_length)
            line1, = axes2[n, 0].plot(obs_error)
            line2, = axes2[n, 1].plot(vecobs_error)
            linegroups.append([line1, line2])
        L = fig2.legend([lines[0] for lines in linegroups], legends)
        make_legend_pickable(L, linegroups)
    fig.savefig("/tmp/dream_comparison_{}.png".format(offset), dpi=100)
    fig2.savefig("/tmp/n_step_error_for_dream_comparison_{}.png".format(offset))
    print("Saved figures.")
#     plt.show()


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
