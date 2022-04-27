import pickle
import numpy as np
from AMP_nav import AMP_nav
from utils.arg_parse import get_parsed_args
import yaml

import wandb



if __name__ == "__main__":
    wandb.init(project="navdream", entity="alan_lanfeng")
    # ===== parse args =====
    args = get_parsed_args()
    with open(f'./cfg/{args.cfg}.yaml') as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

    data1 = np.load('/local/home/lafeng/obs_dict_3_92.npy',allow_pickle=True)
    data2 = np.load('/local/home/lafeng/obs_dict_1000_4000.npy',allow_pickle=True)
    data3 = np.load('/local/home/lafeng/obs_dict_1500_3000.npy', allow_pickle=True)
    data4 = np.load('/local/home/lafeng/obs_dict_4500_6000.npy', allow_pickle=True)

    data = []
    for i in range(6):
        data.append(data1[i])
        data.append(data2[i])
        data.append(data3[i])
        data.append(data4[i])

    real = np.zeros([24,48,64,64,3])
    fake = np.zeros([24, 48, 64, 64, 3])
    for i in range(24):
        real[i]=data[i]['real']
        fake[i] = data[i]['dream']
    real = real.transpose(0,1,4,2,3)
    fake = fake.transpose(0, 1, 4, 2, 3)
    disc = AMP_nav(cfg=cfg)

    disc.load_data(real,fake)

    for i in range(1000):
        loss, info = disc.update_disc()
        wandb.log({"loss": loss})
        wandb.log(info)




