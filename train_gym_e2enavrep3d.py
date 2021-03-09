import numpy as np
from stable_baselines import TD3
from stable_baselines.td3.policies import CnnPolicy
from stable_baselines.ddpg.noise import NormalActionNoise

from navrep3denv import NavRep3DEnv

if __name__ == "__main__":
    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    MODELPATH = "/tmp/model"
    env = NavRep3DEnv(silent=True)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(CnnPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=50000, log_interval=10)
    model.save(MODELPATH)

    del model # remove to demonstrate saving and loading

    model = TD3.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
