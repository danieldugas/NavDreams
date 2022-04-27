
from abc import ABC, abstractmethod
import time
import os
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import numpy as np
from utils.replay_buffer import ReplayBufferRandStorage
from utils.normalizer import Normalizer


class AMP(ABC):

    def __init__(self, cfg=None):
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = cfg['input_dim']

        self.batch_size = cfg['batch_size']

        self._build_model()

        self.expert_data = ReplayBufferRandStorage(cfg['buffer_size'])

        self.agent_data = ReplayBufferRandStorage(cfg['buffer_size'])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                                weight_decay=cfg['weight_decay'], amsgrad=True)

        self.agent_normalizer = Normalizer(self.input_dim)

        self.expert_normalizer = Normalizer(self.input_dim)


    def update_disc(self,obs_expert,obs_agent):
        self.model.train()


        self.optimizer.zero_grad()
        # obs_expert = self.expert_normalizer.normalize(obs_expert)
        # obs_agent = self.agent_normalizer.normalize(obs_agent)
        expert_logit = self.model(obs_expert)
        agent_logit = self.model(obs_agent)

        loss, info = self._compute_loss(expert_logit,agent_logit,obs_expert)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.cpu().detach().numpy(), info


    def _compute_loss(self,expert_logit,agent_logit,obs_expert):
        info = {}

        expert_loss = 0.5 * torch.sum(torch.square(expert_logit - 1), axis=-1)

        agent_loss = 0.5 * torch.sum(torch.square(agent_logit + 1), axis=-1)

        expert_loss = torch.mean(expert_loss)

        agent_loss = torch.mean(agent_loss)

        disc_loss = 0.5*(expert_loss+agent_loss)

        acc_expert = torch.mean(torch.greater(expert_logit, 0).to(float))

        acc_agent = torch.mean(torch.less(agent_logit, 0).to(float))

        # information flow
        state_list = list(self.model.state_dict())
        vars = self.model.state_dict()[state_list[-2]]
        logit_reg_loss = torch.norm(vars,2)
        if torch.isnan(logit_reg_loss):
            logit_reg_loss=0
            print('logit nan!')
        disc_loss += self.cfg['logit_reg_weight']*logit_reg_loss


        # gradient penalty
        grad = torch.autograd.grad(outputs=torch.mean(expert_logit), inputs=obs_expert, create_graph=True)[0]
        grad = torch.mean(torch.sum(torch.square(grad),dim=-1))
        if torch.isnan(grad):
            grad=0
            print('grad nan!')
        disc_loss += self.cfg['grad_penalty']*grad

        info['acc_expert'] = acc_expert.cpu().detach().numpy()
        info['acc_agent'] = acc_agent.cpu().detach().numpy()
        info['expert_loss'] = expert_loss.cpu().detach().numpy()
        info['agent_loss'] = agent_loss.cpu().detach().numpy()

        return disc_loss, info

    def save_model(self):
        model_save_name = os.path.join(self.exp_data_path, 'saved_models',
                                       'model_{}.pt'.format(self.current_epoch))
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch
        }
        torch.save(state, model_save_name)
        self.print('\n model saved to %s' % model_save_name)


    def load_model(self, model_path,device):
        state = torch.load(model_path,map_location=torch.device(device))
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_epoch = state["epoch"]

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def _build_model(self):
        pass
    #
    # @abstractmethod
    # def _build_dataset(self, type):
    #     pass