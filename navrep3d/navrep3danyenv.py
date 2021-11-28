import os

from navrep3d.mlagents_gym_wrapper import NavRep3DStaticASLEnv, NavRep3DStaticASLEnvDiscrete
from navrep3d.archiveenv import ArchiveEnv
from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv, NavRep3DTrainEnvDiscrete

def NavRep3DAnyEnv(**kwargs):
    """ wrapper to hide the difference between the two kinds of navrep3d environments
    (crowdbotchallenge vs mlagents)
    allows creating either a navrep3dtrainenv (train, alt, city, office) or navprep3dstaticasl env
    depending on build name """
    build_name = kwargs.pop('build_name', "staticasl")
    unity_player_dir = kwargs.pop('unity_player_dir', "DEFAULT")
    if build_name == "staticasl":
        if unity_player_dir == "DEFAULT":
            kwargs['unity_player_dir'] = "LFS/executables"
        return NavRep3DStaticASLEnv(**kwargs)
    elif build_name == "rosbag":
        directory = os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")
        return ArchiveEnv(directory, shuffle_episodes=True)
    else:
        return NavRep3DTrainEnv(**kwargs)

def NavRep3DAnyEnvDiscrete(**kwargs):
    """ same as NavRep3DAnyEnv but with DiscreteActionWrapper """
    build_name = kwargs.pop('build_name', "staticasl")
    unity_player_dir = kwargs.pop('unity_player_dir', "DEFAULT")
    if build_name == "staticasl":
        if unity_player_dir == "DEFAULT":
            kwargs['unity_player_dir'] = "LFS/executables"
        return NavRep3DStaticASLEnvDiscrete(**kwargs)
    else:
        return NavRep3DTrainEnvDiscrete(**kwargs)

