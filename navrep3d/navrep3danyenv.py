import os

from navrep3d.mlagents_gym_wrapper import NavRep3DStaticASLEnv, NavRep3DStaticASLEnvDiscrete
from navrep3d.archiveenv import ArchiveEnv
from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv, NavRep3DTrainEnvDiscrete
from navrep3d.custom_policy import FlattenN3DObsWrapper

def NavRep3DAnyEnv(**kwargs):
    """ wrapper to hide the difference between the two kinds of navrep3d environments
    (crowdbotchallenge vs mlagents)
    allows creating either a navrep3dtrainenv (train, alt, city, office) or navprep3dstaticasl env
    depending on build name """
    build_name = kwargs.get('build_name', "./build.x86_64")
    if build_name in ["staticasl", "cathedral", "gallery"]:
        return NavRep3DStaticASLEnv(**kwargs)
    elif build_name == "rosbag":
        directory = os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")
        return ArchiveEnv(directory, shuffle_episodes=True)
    else:
        return NavRep3DTrainEnv(**kwargs)

def NavRep3DAnyEnvDiscrete(**kwargs):
    """ same as NavRep3DAnyEnv but with DiscreteActionWrapper """
    build_name = kwargs.get('build_name', "./build.x86_64")
    if build_name in ["staticasl", "cathedral", "gallery"]:
        return NavRep3DStaticASLEnvDiscrete(**kwargs)
    elif build_name == "rosbag":
        raise NotImplementedError("No known uses for achive env with discrete action")
    else:
        return NavRep3DTrainEnvDiscrete(**kwargs)

def NavRep3DAnyEnvDiscreteFlattened(**kwargs):
    """ same as NavRep3DAnyEnv but with DiscreteActionWrapper
    and FlattenN3DObsWrapper (used by custom E2E policy) """
    return FlattenN3DObsWrapper(NavRep3DAnyEnvDiscrete(**kwargs))


if __name__ == "__main__":
    for build_name in [
            "./build.x86_64",
            "./alternate.x86_64",
            "./city.x86_64",
            "./office.x86_64",
            "staticasl",
            "rosbag",
    ]:
        for envtype in [NavRep3DAnyEnv, NavRep3DAnyEnvDiscrete]:
            if build_name == "rosbag" and envtype == NavRep3DAnyEnvDiscrete:
                continue
            env = envtype(build_name=build_name)
            print(build_name)
            print(env.reset())
            print(env.step(env.action_space.sample()))
            env.render()
            env.close()
