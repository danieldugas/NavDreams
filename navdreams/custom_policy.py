import numpy as np
import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from navdreams.navrep3dtrainenv import NavRep3DTrainEnv, NavRep3DTrainEnvDiscrete

_RS = 5
_64 = 64
_CH = 3

class NavRep3DTrainEnvFlattened(NavRep3DTrainEnv):
    # returns only the robotstate as obs
    def __init__(self, *args, **kwargs):
        super(NavRep3DTrainEnvFlattened, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            np.prod(self.observation_space[0].shape)+self.observation_space[1].shape[0],), dtype=np.float32)

    def step(self, actions):
        obs, reward, done, info = super(NavRep3DTrainEnvFlattened, self).step(actions)
        # image: channels first, normalized flattened. vector: same
        obs = np.concatenate([(np.moveaxis(obs[0], -1, 0) / 255.).flatten(), (obs[1]).flatten()], axis=0)
        return obs, reward, done, info

class NavRep3DTrainEnvDiscreteFlattened(NavRep3DTrainEnvDiscrete):
    # returns only the robotstate as obs
    def __init__(self, *args, **kwargs):
        super(NavRep3DTrainEnvDiscreteFlattened, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            np.prod(self.observation_space[0].shape)+self.observation_space[1].shape[0],), dtype=np.float32)

    def step(self, actions):
        obs, reward, done, info = super(NavRep3DTrainEnvDiscreteFlattened, self).step(actions)
        # image: channels first, normalized flattened. vector: same
        obs = np.concatenate([(np.moveaxis(obs[0], -1, 0) / 255.).flatten(), (obs[1]).flatten()], axis=0)
        return obs, reward, done, info

class FlattenN3DObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenN3DObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            np.prod(env.observation_space[0].shape)+env.observation_space[1].shape[0],), dtype=np.float32)

    def observation(self, observation):
        # image: channels first, normalized flattened. vector: same
        flatobs = np.concatenate([(np.moveaxis(observation[0], -1, 0) / 255.).flatten(),
                                  (observation[1]).flatten()], axis=0)
        return flatobs

class NavRep3DTupleCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 cnn_features_dim: int = 256, mlp_features_dim: int = 16):
        super(NavRep3DTupleCNN, self).__init__(observation_space, cnn_features_dim + mlp_features_dim)
        # hardcoded, because running into issues with dict observation_spaces
        self.image_shape = (_CH,_64,_64)
        self.image_shape_flat = np.prod(self.image_shape)
        self.vector_shape_flat = _RS
        # observation[0] is image
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = _CH
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = observation_space.sample()
            sample = sample[:self.image_shape_flat]
            sample = np.reshape(sample, (1,) + self.image_shape) # add batch dim
            n_flatten = self.cnn(
                th.as_tensor(sample).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_features_dim), nn.ReLU())

        # observation[1] is vector
        self.mlp = nn.Linear(self.vector_shape_flat, mlp_features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image = th.reshape(observations[:, :self.image_shape_flat], (-1,) + self.image_shape)
        vector = observations[:, -self.vector_shape_flat:]
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat([self.linear(self.cnn(image)), self.mlp(vector)], dim=1)


if __name__ == "__main__":
    policy_kwargs = dict(
        features_extractor_class=NavRep3DTupleCNN,
        features_extractor_kwargs=dict(cnn_features_dim=64),
    )
    model = PPO("CnnPolicy", "NavRep3DTrainEnvFlattened", policy_kwargs=policy_kwargs, verbose=1)
    model.learn(1000)
