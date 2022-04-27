from AMP import AMP
import torch.nn as nn
from model.single_image_disc import imageDisc

class AMP_nav(AMP):

    def __init__(self,cfg=None):
        super().__init__(cfg=cfg)

    def load_data(self,agent_data, expert_data):

        self.agent_data.store(agent_data)
        self.expert_data.store(expert_data)

    def _build_model(self):
        self.model = imageDisc().to(self.device)

