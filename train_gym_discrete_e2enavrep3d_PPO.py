import os
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from strictfire import StrictFire

from navrep3d.sb3_callbacks import NavRep3DLogCallback
from navrep3d.custom_policy import NavRep3DTupleCNN
from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscreteFlattened
from navrep3d.auto_debug import enable_auto_debug

MILLION = 1000000

def main(dry_run=False, n=None, build_name=None):
    TRAIN_STEPS = n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 5 * MILLION
    _C = 64
    DIR = os.path.expanduser("~/navrep3d/models/gym")
    LOGDIR = os.path.expanduser("~/navrep3d/logs/gym")
    if dry_run:
        DIR = "/tmp/navrep3d/models/gym"
        LOGDIR = "/tmp/navrep3d/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    CONTROLLER_ARCH = "_VCARCH_C{}".format(_C)
    if build_name is None:
        build_name = "./alternate.x86_64"
    build_names = [build_name] * 4
    if build_name == "./build.x86_64":
        ENV_NAME = "navrep3dtrainenv_"
    elif build_name == "./alternate.x86_64":
        ENV_NAME = "navrep3daltenv_"
    elif build_name == "SC":
        ENV_NAME = "navrep3dSCenv_"
        build_names = ["./alternate.x86_64", "./city.x86_64", "./office.x86_64", "./office.x86_64"]
    elif build_name == "SCR":
        ENV_NAME = "navrep3dSCRfixedenv_"
        build_names = ["./alternate.x86_64", "./city.x86_64", "./office.x86_64", "staticasl"]
    elif build_name == "staticasl":
        ENV_NAME = "navrep3daslfixedenv_"
    else:
        raise NotImplementedError
    LOGNAME = ENV_NAME + START_TIME + "_DISCRETE_PPO" + "_E2E" + CONTROLLER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = SubprocVecEnv([
        lambda: NavRep3DAnyEnvDiscreteFlattened(build_name=build_names[0], debug_export_every_n_episodes=170, port=25002),
        lambda: NavRep3DAnyEnvDiscreteFlattened(build_name=build_names[1], debug_export_every_n_episodes=0, port=25003),
        lambda: NavRep3DAnyEnvDiscreteFlattened(build_name=build_names[2], debug_export_every_n_episodes=0, port=25004),
        lambda: NavRep3DAnyEnvDiscreteFlattened(build_name=build_names[3], debug_export_every_n_episodes=0, port=25005),
    ])

    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)

    policy_kwargs = dict(
        features_extractor_class=NavRep3DTupleCNN,
        features_extractor_kwargs=dict(cnn_features_dim=_C),
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)

    if False:
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

    env.close()


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
