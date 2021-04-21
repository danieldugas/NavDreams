import os
from navrep.tools.commonargs import parse_common_args
from datetime import datetime
from stable_baselines3 import PPO
import numpy as np
import gym

from sb3_callbacks import NavRep3DLogCallback
from navrep3denv import NavRep3DEnv

np.set_printoptions(suppress=True, precision=2)

class RSNavRep3DEnv(NavRep3DEnv):
    # returns only the robotstate as obs
    def __init__(self, *args, **kwargs):
        super(RSNavRep3DEnv, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def step(self, actions):
        obs, reward, done, info = super(RSNavRep3DEnv, self).step(actions)
        obs = obs[1]
        print(obs)
        # detect fd up situation after reset
        if np.allclose(actions, np.array([0,0,0])) and np.any(np.abs(obs) > 100.):
            print("WTF")
        return obs, reward, done, info


if __name__ == "__main__":
    args, _ = parse_common_args()

    # this is a debug script after all
    args.dry_run = True

    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    DIR = os.path.expanduser("~/navrep3d/models/gym")
    LOGDIR = os.path.expanduser("~/navrep3d/logs/gym")
    if args.dry_run:
        DIR = "/tmp/navrep3d/models/gym"
        LOGDIR = "/tmp/navrep3d/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    CONTROLLER_ARCH = "_VCARCH_C64"
    LOGNAME = "rsnavrep3dtrainenv_" + START_TIME + "_PPO" + "_E2E" + CONTROLLER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    env = RSNavRep3DEnv(verbose=1)

    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)

    print("Saving model to {}".format(MODELPATH))
    model.save(MODELPATH)

    del model # remove to demonstrate saving and loading

    model = PPO.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        env.render()
