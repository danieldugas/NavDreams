import os
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navrep3d.rssm import RSSMWMConf, RSSMWorldModel

def main(dataset="SCR",
         gpu=False,
         dream_length=16,
         context_length=16,
         n_examples=5,
         ):
    sequence_length = dream_length + context_length

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
        examples = [34, 51, 23, 42, 79, 5, 120]
    elif dataset == "staticasl":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl")]
        examples = [34, 51, 23, 42, 79, 5, 120]
    else:
        raise NotImplementedError(dataset)

    worldmodel_types = ["Transformer", "RSSM_A0"]
    worldmodels = []
    for worldmodel_type in worldmodel_types:
        if worldmodel_type == "Transformer":
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
        elif worldmodel_type == "RSSM_A0":
            wm_model_path = "~/navrep3d_W/models/W/RSSM_A0_{}".format(dataset)
            wm_model_path = os.path.expanduser(wm_model_path)
            mconf = RSSMWMConf()
            mconf.image_channels = 3
            model = RSSMWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            worldmodel = model
        worldmodels.append(worldmodel)

    example_sequences = {examples[i]: None for i in range(n_examples)}
    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=64)
    for i, (x, a, y, x_rs, y_rs, dones) in enumerate(seq_loader):
        if i in example_sequences:
            example_sequences[i] = (x, a, y, x_rs, y_rs, dones)
        if i > max(examples):
            break

    n_rows_per_example = (len(worldmodels) + 1)
    fig, axes = plt.subplots(n_rows_per_example * n_examples, sequence_length, num="dream")
    for n, id_ in enumerate(example_sequences):
        if example_sequences[id_] is None:
            continue
        x, a, y, x_rs, y_rs, dones = example_sequences[id_]
        real_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i]) for i in range(sequence_length)]
        dream_sequences = []
        for worldmodel in worldmodels:
            dream_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i]) for i in range(context_length)]
            dream_sequence = dream_sequence[:context_length]
            dream_sequence[-1]['action'] = None
            next_actions = a[context_length-1:sequence_length]
            for action in next_actions:
                dream_sequence[-1]['action'] = action * 1.
                img_npred, goal_pred = worldmodel.get_next(dream_sequence)
                # update sequence
                dream_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
            dream_sequences.append(dream_sequence)

        for i in range(sequence_length):
            axes[n_rows_per_example*n, i].imshow(real_sequence[i]['obs'])
            axes[n_rows_per_example*n, i].set_ylabel("GT")
            for m, dream_sequence in enumerate(dream_sequences):
                axes[n_rows_per_example*n+1+m, i].set_ylabel("{}".format(worldmodel_types[m]))
                axes[n_rows_per_example*n+1+m, i].imshow(dream_sequence[i]['obs'])
    plt.show()


if __name__ == "__main__":
    StrictFire(main())
