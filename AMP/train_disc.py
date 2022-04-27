import pickle
import numpy as np
from AMP_nav import AMP_nav
from utils.arg_parse import get_parsed_args
import yaml
import pickle
import wandb
import os
import torch

if __name__ == "__main__":
    a = torch.randn(64, 100, 1, 1)
    # ===== parse args =====
    args = get_parsed_args()
    wandb.init(project="navdream", entity="alan_lanfeng",name = args.exp_name)

    with open(f'./cfg/{args.cfg}.yaml') as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    disc = AMP_nav(cfg=cfg)

    data_len = cfg['data_size']
    test_len = cfg['test_size']
    train_len = data_len-test_len
    data_dir = cfg['data_path']
    bs = cfg['batch_size']

    epoch_step_num = int(train_len/bs*2)

    test_step_num = int(np.floor(test_len/bs))

    for epoch in range(100):

        for i in range(epoch_step_num):
            idx = np.random.randint(0, train_len, size=bs)
            obs_agent = np.zeros([bs,64,3,64,64])
            obs_expert = np.zeros([bs, 64, 3, 64, 64])

            for j in range(len(idx)):
                path = os.path.join(data_dir, f'{idx[j]}.pkl')
                with open(path, "rb+") as f:
                    data = pickle.load(f)
                real = data['real'][np.newaxis, ...]
                fake = data['fake'][np.newaxis, ...]
                obs_expert[j] = real.transpose(0, 1, 4, 2, 3)
                obs_agent[j] = fake.transpose(0, 1, 4, 2, 3)

            obs_expert = torch.Tensor(obs_expert).to(disc.device)
            obs_expert.requires_grad = True
            obs_agent = torch.Tensor(obs_agent).to(disc.device)
            loss, info = disc.update_disc(obs_expert,obs_agent)
            #wandb.log({"loss": loss})
            wandb.log(info)

        print(f'epoch: {epoch} finished, evaluating..')

        acc_agent_test = np.zeros(test_step_num)
        acc_expert_test = np.zeros(test_step_num)

        conf_agent = np.zeros(test_step_num)
        conf_expert = np.zeros(test_step_num)

        disc.model.eval()
        for i in range(test_step_num):
            idx = list(range(train_len+i,train_len+i+bs))
            obs_agent = np.zeros([bs, 64, 3, 64, 64])
            obs_expert = np.zeros([bs, 64, 3, 64, 64])

            for j in range(len(idx)):
                path = os.path.join(data_dir, f'{idx[j]}.pkl')
                with open(path, "rb+") as f:
                    data = pickle.load(f)
                real = data['real'][np.newaxis, ...]
                fake = data['fake'][np.newaxis, ...]
                obs_expert[j] = real.transpose(0, 1, 4, 2, 3)
                obs_agent[j] = fake.transpose(0, 1, 4, 2, 3)

            obs_expert = torch.Tensor(obs_expert).to(disc.device)
            obs_agent = torch.Tensor(obs_agent).to(disc.device)

            expert_logit = disc.model(obs_expert)
            agent_logit = disc.model(obs_agent)

            acc_expert_test[i] = torch.mean(torch.greater(expert_logit, 0).to(float))
            acc_agent_test[i] = torch.mean(torch.less(agent_logit, 0).to(float))
            conf_expert[i] = torch.mean(expert_logit)
            conf_agent[i] = torch.mean(agent_logit)
            if epoch>5:
                mistake_expert = torch.where(expert_logit<0)[0]+i*bs
                mistake_agent = torch.where(agent_logit>0)[0]+i*bs
                print(f'agent mis: {mistake_agent}')
                print(f'expert mis: {mistake_expert}')

        test_info = {}
        test_info['acc_expert_test'] = np.mean(acc_expert_test)
        test_info['acc_agent_test'] = np.mean(acc_agent_test)
        test_info['confident_expert'] = np.mean(conf_expert)
        test_info['confident_agent'] = np.mean(conf_agent)
        wandb.log(test_info)






