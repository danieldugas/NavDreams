from matplotlib import pyplot as plt
import os
import numpy as np
from strictfire import StrictFire

from plot_gym_training_progress import make_legend_pickable

def dreamertrainenv_to_n3dtrainenv(trainenv):
    if trainenv == "NavRep3DStaticASLEnv":
        envname = "navrep3daslfixedenv"
    elif trainenv == "NavRep3DKozeHDEnv":
        envname = "navrep3dkozehdrenv"
    elif trainenv == "NavRep3DKozeHDRSEnv":
        envname = "navrep3dkozehdrsenv"
    elif trainenv == "NavRep3DTrainEnv":
        envname = "navrep3daltenv"
    else:
        raise NotImplementedError
    return envname

def info_from_filename(filename):
    # filename: "env_date_MODELINFO_variant_ckpt_build_difficulty_N.npz"
    rest = filename
    rest = rest.replace(".npz", "")
    rest, n_episodes = rest.rsplit("_", 1)
    rest, difficulty = rest.rsplit("_", 1)
    rest, build = rest.rsplit("_", 1)
    trainenv, rest = rest.split("_", 1)
    # rest is now "date/id_MODELINFO
    if "DREAMER" in rest:
        mtype = "DREAMER"
        # rest is "id_DREAMER"
        uid, rest = rest.split("_", 1)
        ckpt = "ckpt"
        wmscope = None
        wmtype = "RSSM"
        trainenv = dreamertrainenv_to_n3dtrainenv(trainenv)
    elif "E2E" in rest:
        mtype = "E2E"
        # rest is "date_DISCRETE_PPO_E2E_VCARCH_C64_ckpt"
        rest, ckpt = rest.rsplit("_", 1)
        wmscope = None
        uid, rest = rest.split("_DISCRETE_PPO_", 1)
        wmtype = None
    else:
        mtype = "N3D"
        # rest is "date_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt"
        rest, ckpt = rest.rsplit("_", 1)
        rest, wmscope = rest.rsplit("_", 1)
        uid, rest = rest.split("_DISCRETE_PPO_", 1)
        wmtype, rest = rest.split("_", 1)
    # postprocess
    trainenv = trainenv.replace("encoded", "")
    n_episodes = int(n_episodes)
    return build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid

def compare_lookup_and_key(lookup, key):
    assert len(lookup) == len(key)
    for a, b in zip(lookup, key):
        if a is None or b is None:
            continue
        if a != b:
            return False
    return True

def find_matches_in_data(lookup, data, alert_if_not_found=False, alert_if_several=False):
    # find data
    found = 0
    matches = []
    for key in data:
        if compare_lookup_and_key(lookup, key):
            found += 1
            matches.append(data[key])
    if alert_if_not_found:
        if found == 0:
            print_diffs(lookup, data)
            raise ValueError("not found")
    if alert_if_several:
        print(lookup)
        if found > 1:
            print_matches(lookup, data)
            raise ValueError("several matches found")
    return matches

def diff_lookup_and_key(lookup, key):
    assert len(lookup) == len(key)
    diff_l = []
    diff_k = []
    for a, b in zip(lookup, key):
        if a is None or b is None:
            diff_l.append(None)
            diff_k.append(None)
            continue
        if a != b:
            diff_l.append(a)
            diff_k.append(b)
        else:
            diff_l.append(None)
            diff_k.append(None)
    return diff_l, diff_k

def print_diffs(lookup, data):
    print(lookup)
    print("")
    print("not found in data. differences:")
    print("")
    # find data
    for key in data:
        diffl, diffk = diff_lookup_and_key(lookup, key)
        print(diffk)

def print_matches(lookup, data):
    print(lookup)
    print("")
    for key in data:
        if compare_lookup_and_key(lookup, key):
            print(key)

def main(
    logdir="~/navrep3d/test/",
):
    # find logfiles
    logdir = os.path.expanduser(logdir)
#     logdir = os.path.join(logdir, "")
    logs = sorted(os.listdir(logdir))

    # extract data from files
    data = {}
    for log in logs:
        path = os.path.join(logdir, log)
        info = info_from_filename(log)
        build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = info

        arrays = np.load(path)
        data[info] = arrays

    # single plot with all gallery
    any_ = None

    fig, axes = plt.subplots(2, 1, num="xtest")

    # xtest in gallery
    N = 10
    bar_lookups = [
        ("alternate", "N3D", any_, "hardest", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("staticasl", "N3D", any_, "medium", "navrep3daslfixedenv", N, "SCR", "GPT", any_),
        ("kozehd", "N3D", any_, "easiest", "navrep3dkozehdrsenv", N, "K2", "GPT", any_),
        ("kozehd", "N3D", any_, "easy", "navrep3dkozehdrsenv", N, "K2", "GPT", any_),
        ("alternate", "DREAMER", any_, "hardest", "navrep3daltenv", N, any_, "RSSM", any_),
        ("staticasl", "DREAMER", any_, "medium", "navrep3daslfixedenv", N, any_, "RSSM", any_),
        ("kozehd", "DREAMER", any_, "easiest", "navrep3dkozehdrsenv", N, any_, "RSSM", any_),
        ("kozehd", "DREAMER", any_, "easy", "navrep3dkozehdrsenv", N, any_, "RSSM", any_),
    ]
    values = []
    crashes = []
    crashesother = []
    timeouts = []
    labels = []
    for lookup in bar_lookups:
        matches = find_matches_in_data(lookup, data, alert_if_not_found=True)
        assert len(matches) == 1
        arrays = matches[0]
        # values
        successes = arrays["successes"]
        difficulties = arrays["difficulties"]
        causes = arrays["causes"]
        lengths = arrays["lengths"]
        # if you recompute timeouts, need to overwrite other causes
        crashes.append(np.mean(causes == "Collision"))
        crashesother.append(np.mean(causes == "Collision from other agent"))
        values.append(np.mean(successes))
        # label
        build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = lookup
        labels.append(str((mtype, trainenv)))
    ax = axes[0]
    ax.bar(labels, values)
    ax.bar(labels, crashes, bottom=values)
    ax.bar(labels, crashesother, bottom=np.array(values)+np.array(crashes))

    plt.show()

    # xtest in gallery
    N = 100
    bar_lookups = [
        ("gallery", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("gallery", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
        ("gallery", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ("gallery", "E2E", any_, "easy", "navrep3daltenv", N, any_, any_, any_),
        ("gallery", "E2E", any_, "easy", "navrep3dSCenv", N, any_, any_, any_),
        ("gallery", "E2E", any_, "easy", "navrep3dSCRenv", N, any_, any_, any_),
    ]
    values = []
    labels = []
    for lookup in bar_lookups:
        matches = find_matches_in_data(lookup, data, alert_if_not_found=True)
        assert len(matches) == 1
        arrays = matches[0]
        # values
        successes = arrays["successes"]
        difficulties = arrays["difficulties"]
        values.append(np.mean(successes))
        # label
        build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = lookup
        labels.append(str((mtype, trainenv)))
    ax = axes[0]
    ax.bar(labels, values)

    plt.show()

    # xtest in cathedral
    N = 100
    bar_lookups = [
        ("cathedral", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("cathedral", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
        ("cathedral", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ("cathedral", "E2E", any_, "easy", "navrep3daltenv", N, any_, any_, any_),
        ("cathedral", "E2E", any_, "easy", "navrep3dSCenv", N, any_, any_, any_),
        ("cathedral", "E2E", any_, "easy", "navrep3dSCRenv", N, any_, any_, any_),
    ]
    values = []
    labels = []
    for lookup in bar_lookups:
        matches = find_matches_in_data(lookup, data, alert_if_not_found=True)
        assert len(matches) == 1
        arrays = matches[0]
        # values
        successes = arrays["successes"]
        difficulties = arrays["difficulties"]
        values.append(np.mean(successes))
        # label
        build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = lookup
        labels.append(str((mtype, trainenv)))
    ax = axes[1]
    ax.bar(labels, values)

    plt.show()


if __name__ == "__main__":
    StrictFire(main)
