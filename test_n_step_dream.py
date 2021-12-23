import os
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navrep3d.rssm import RSSMWMConf, RSSMWorldModel

def main(dataset="SCR",
         worldmodel_type="Transformer",
         gpu=False,
         dream_length=16,
         context_length=16,
         wm_model_path=None
         ):
    sequence_length = dream_length + context_length

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
    elif dataset == "staticasl":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl")]
    else:
        raise NotImplementedError(dataset)

    if worldmodel_type == "Transformer":
        if wm_model_path is None:
            wm_model_path = "~/navrep3d_W/models/W/transformer_SCR"
        wm_model_path = os.path.expanduser(wm_model_path)
        BLOCK_SIZE = 32
        _H = 64
        _C = 3
        mconf = GPTConfig(BLOCK_SIZE, _H)
        mconf.image_channels = _C
        model = GPT(mconf, gpu=gpu)
        load_checkpoint(model, wm_model_path, gpu=gpu)
        worldmodel = model
    elif worldmodel_type == "RSSM":
        if wm_model_path is None:
            wm_model_path = "~/navrep3d_W/models/W/RSSM_A0_SCR"
        wm_model_path = os.path.expanduser(wm_model_path)
        mconf = RSSMWMConf()
        mconf.image_channels = 3
        model = RSSMWorldModel(mconf, gpu=gpu)
        load_checkpoint(model, wm_model_path, gpu=gpu)
        worldmodel = model

    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=64)
    for x, a, y, x_rs, y_rs, dones in seq_loader:
        break
    real_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i]) for i in range(sequence_length)]
    gpt_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i]) for i in range(context_length)]
    gpt_sequence = gpt_sequence[:context_length]
    gpt_sequence[-1]['action'] = None
    next_actions = a[context_length-1:sequence_length]
    for action in next_actions:
        gpt_sequence[-1]['action'] = action * 1.
        img_npred, goal_pred = worldmodel.get_next(gpt_sequence)
        # update sequence
        gpt_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))

    fig, axes = plt.subplots(2, sequence_length, num="dream")
    for i in range(sequence_length):
        axes[0, i].imshow(real_sequence[i]['obs'])
        axes[1, i].imshow(gpt_sequence[i]['obs'])
    plt.show()


if __name__ == "__main__":
    StrictFire(main())

