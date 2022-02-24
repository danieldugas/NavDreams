from matplotlib import pyplot as plt
import os
import numpy as np
from strictfire import StrictFire

# from plot_gym_training_progress import make_legend_pickable

scenario_paper_names = {
    "alternate": "simple",
    "city": "city",
    "office": "office",
    "staticasl": "modern",
    "cathedral": "cathedral",
    "gallery": "gallery",
    "kozehd": "sim2real",
}

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
    keys = []
    for key in data:
        if compare_lookup_and_key(lookup, key):
            found += 1
            matches.append(data[key])
            keys.append(key)
    if alert_if_not_found:
        if found == 0:
            print_diffs(lookup, data)
            raise ValueError("not found")
    if alert_if_several:
        print(lookup)
        if found > 1:
            print_matches(lookup, data)
            raise ValueError("several matches found")
    return matches, keys

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
    paper=False,
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

    any_ = None

    def to_bar_chart(bar_lookups, ax, labels=None, hide_error=False):
        values = []
        asy_errors = []
        crashes = []
        crashesother = []
        timeouts = []
        foundkeys = []
        for lookup in bar_lookups:
            matches, keys = find_matches_in_data(lookup, data, alert_if_not_found=True)
            if len(matches) != 1:
                raise ValueError("Matches != 1:\nfor\n{}\nfound\n{}".format(lookup, keys))
            arrays = matches[0]
            key = keys[0]
            # values
            successes = arrays["successes"]
#             difficulties = arrays["difficulties"]
            causes = arrays["causes"]
            lengths = arrays["lengths"]
            # if you recompute timeouts, need to overwrite other causes
            recalc_timeout = None
            recalc_timeout = int(180. / 0.2)
            if recalc_timeout is not None:
                causes[lengths > recalc_timeout] = "Timeout"
                successes[lengths > recalc_timeout] = 0
            timeouts.append(np.mean(causes == "Timeout"))
            crashes.append(np.mean(causes == "Collision"))
            crashesother.append(np.mean(causes == "Collision from other agent"))
            values.append(np.mean(successes))
            # spread is difference between mean of first half and second half
            splits = np.array_split(successes, 2)
            splits = [np.mean(s) for s in splits]
            asy_error = [abs(min(splits)-np.mean(successes)), abs(max(splits)-np.mean(successes))]
            asy_errors.append(asy_error)
            # label
            build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = lookup
            foundkeys.append(key)
        if labels is None:
            labels = [str(k) for k in foundkeys]

        values = np.array(values)
        asy_errors = np.array(asy_errors).reshape((len(values), 2)).T
        timeouts = np.array(timeouts)
        crashes = np.array(crashes)
        crashesother = np.array(crashesother)
        if hide_error:
            asy_errors = None
        ax.bar(labels, values, yerr=asy_errors, color="mediumseagreen")
        ax.bar(labels, timeouts, bottom=values, color="lightgrey")
        ax.bar(labels, crashes, bottom=values+timeouts, color="orange")
        ax.bar(labels, crashesother, bottom=values+timeouts+crashes, color="tomato")

    if not paper:
        # all plots
        all_builds = sorted(list(set([(key[0], key[3]) for key in data])))
        all_mtypes = sorted(list(set([key[1] for key in data])))
        cols = len(all_mtypes)
        rows = len(all_builds)
        fig, axes = plt.subplots(rows, cols, num="all_tests")
        axes = np.array(axes).reshape((rows, cols))
        for row, (build, diff) in enumerate(all_builds):
            for col, mtype in enumerate(all_mtypes):
                ax = axes[row, col]
                bar_lookups = [key for key in data
                               if key[0] == build and key[3] == diff and key[1] == mtype]
                bar_lookups = [key for key in bar_lookups if key[5] >= 50]
                to_bar_chart(bar_lookups, ax)
                if col == 0:
                    ax.set_ylabel("{}\n{}".format(build, diff))
                if row == 0:
                    ax.set_title(mtype)
        plt.show()

    # single plot with best in each
    ROT = True
    N = 100
    pairs = [
        [
            ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", "2021_12_06__21_45_47"), # noqa
            ("alternate", "E2E", any_, "hardest", "navrep3daltenv", N, any_, any_, "2021_11_01__08_52_03"), # noqa
        ], [
            ("city", "N3D", "bestckpt", "hardest", "navrep3dcityenv", N, "SCR", "GPT", "2022_02_18__18_26_31"), # noqa
            ("city", "E2E", any_, "hardest", "navrep3dcityenv", N, any_, any_, "2022_02_19__16_34_05"), # noqa
        ], [
            ("office", "N3D", "bestckpt", "random", "navrep3dofficeenv", N, "SCR", "GPT", "2022_02_19__16_33_28"), # noqa
            ("office", "E2E", any_, "random", "navrep3dofficeenv", N, any_, any_, "2022_02_17__21_27_47"), # noqa
        ], [
            ("staticasl", "N3D", "bestckpt", "medium", "navrep3daslfixedenv", N, "SCR", "GPT", "2021_12_29__17_17_16"), # noqa
            ("staticasl", "E2E", any_, "medium", "navrep3daslfixedenv", N, any_, any_, "2022_01_01__13_09_23"), # noqa
        ], [
            ("cathedral", "N3D", "bestckpt", "medium", "navrep3dcathedralenv", N, "SCR", "GPT", "2022_02_14__10_22_45"), # noqa
            ("cathedral", "E2E", any_, "medium", "navrep3dcathedralenv", N, any_, any_, "2022_02_11__18_09_16"), # noqa
        ], [
            ("gallery", "N3D", "bestckpt", "easy", "navrep3dgalleryenv", N, "SCR", "GPT", "2022_02_11__21_52_34"), # noqa
            ("gallery", "E2E", any_, "easy", "navrep3dgalleryenv", N, any_, any_, "2022_02_16__15_08_38"), # noqa
        ], [
            ("kozehd", "N3D", "bestckpt", "easiest", "navrep3dkozehdrsenv", N, "K2", "GPT", "2022_02_02__17_18_59"), # noqa
            ("kozehd", "E2E", any_, "easiest", "navrep3dkozehdrsenv", N, any_, any_, "2022_02_06__22_58_00"), # noqa
        ], [
            ("kozehd", "N3D", "bestckpt", "easy", "navrep3dkozehdrsenv", N, "K2", "GPT", "2022_02_02__17_18_59"), # noqa
            ("kozehd", "E2E", any_, "easy", "navrep3dkozehdrsenv", N, any_, any_, "2022_02_06__22_58_00"), # noqa
        ]
    ]
    rows = len(pairs)
    cols = 1
    fig, axes = plt.subplots(rows, cols, num=("best_test_rot" if ROT else "best_test"))
    axes = np.array(axes).reshape((rows, cols))
#     labels = [lookup[1] for lookup in bar_lookups]
    labels = None
#         "Dreamer",
    if ROT:
        pairs = pairs[::-1]
    for row in range(rows):
        col = 0
        ax = axes[row, col]
        bar_lookups = pairs[row]
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper)
        ax.set_xticklabels(["", ""])
        name = scenario_paper_names[bar_lookups[0][0]]
        name = name + "\n(empty)" if bar_lookups[0][3] == "easiest" else name
        ax.set_ylabel(name)
        if ROT:
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
    ax.set_xticklabels(["World-model", "End-to-end"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()

    # single plot with best in simple
    fig, axes = plt.subplots(1, 1, num="simple")
    N = 100
    bar_lookups = [
        ("alternate", "DREAMER", "ckpt", "hardest", "navrep3daltenv", N, any_, "RSSM", any_),
        ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", "2021_12_06__21_45_47"),
        ("alternate", "E2E", any_, "hardest", "navrep3daltenv", N, any_, any_, "2021_11_01__08_52_03"),
    ]
#         ("alternate", "DREAMER", any_, "hardest", "navrep3daltenv", N, any_, any_, any_),
#     labels = [lookup[1] for lookup in bar_lookups]
    labels = [
        "Dreamer",
        "World-model",
        "End-to-end",
    ]
#         "Dreamer",
    ax = axes
    to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()

    # benefit of generalization
    # single plot with OOD in cathedral, gallery, kozehd
    ROT = True
    N = 100
    pairs = [
        [
            ("cathedral", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
            ("cathedral", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
            ("cathedral", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
            ("cathedral", "N3D", any_, "easy", "navrep3dcathedralenv", N, "SCR", "GPT", any_),
        ], [
            ("gallery", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
            ("gallery", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
            ("gallery", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
            ("gallery", "N3D", any_, "easy", "navrep3dgalleryenv", N, "SCR", "GPT", any_),
        ], [
            ("kozehd", "N3D", any_, "easiest", "navrep3daltenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easiest", "navrep3dSCenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easiest", "navrep3dSCRenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easiest", "navrep3dkozehdrsenv", N, "K2", "GPT", any_),
        ]
    ]
    figure_mosaic = """
    AAAB
    CCCD
    EEEF
    """
    fig, axes = plt.subplot_mosaic(figure_mosaic, num=("ood_cgk"))
    axes = [axes[letter] for letter in 
            sorted(list(set(figure_mosaic.replace("\n","").replace(" ", ""))))]
    rows = 3
    cols = 2
#     rows = len(pairs)
#     cols = 2
#     fig, axes = plt.subplots(rows, cols, num=("ood_cgk"))
    axes = np.array(axes).reshape((rows, cols))
#     labels = [lookup[1] for lookup in bar_lookups]
    labels = None
#         "Dreamer",
    if ROT:
        pairs = pairs[::-1]
    for row in range(rows):
        col = 0
        ax = axes[row, col]
        bar_lookups = pairs[row][:3]
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper)
        ax.set_xticklabels(["", ""])
        name = scenario_paper_names[bar_lookups[0][0]]
        name = name + "\n(empty)" if bar_lookups[0][3] == "easiest" else name
        name = name + "\n(sparse)" if bar_lookups[0][0] == "cathedral" and bar_lookups[0][3] == "easy" else name
        ax.set_ylabel(name)
        ax.set_ylim([0, 1.05])
        if ROT:
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
        col = 1
        ax = axes[row, col]
        bar_lookups = pairs[row][-1:]
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper)
        ax.set_xticklabels(["", ""])
        ax.set_yticks([], [])
        ax.set_ylim([0, 1.05])
    axes[-1, 0].set_xticklabels(["S", "SC", "SCR"])
    axes[-1, 1].set_xticklabels(["Domain Specific (for Reference)"])
    for tick in axes[-1, 0].get_xticklabels():
        tick.set_rotation(90)
    for tick in axes[-1, 1].get_xticklabels():
        tick.set_rotation(90)

    # cost of generalization
    # single plot with generalists in simple, city, office, alt
    ROT = True
    N = 100
    pairs = [
        [
            ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", "2021_12_06__21_45_47"), # noqa
            ("alternate", "N3D", any_, "hardest", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [
            ("city", "N3D", "bestckpt", "hardest", "navrep3dcityenv", N, "SCR", "GPT", "2022_02_18__18_26_31"), # noqa
            ("city", "N3D", any_, "hardest", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [
            ("office", "N3D", "bestckpt", "random", "navrep3dofficeenv", N, "SCR", "GPT", "2022_02_19__16_33_28"), # noqa
            ("office", "N3D", any_, "random", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [ # this one is wrong! training in old but testing in fixed env
            ("staticasl", "N3D", "bestckpt", "medium", "navrep3daslfixedenv", N, "SCR", "GPT", "2021_12_29__17_17_16"), # noqa
            ("staticasl", "N3D", any_, "medium", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ]
    ]
    rows = len(pairs)
    cols = 1
    fig, axes = plt.subplots(rows, cols, num=("generalist_cost"))
    axes = np.array(axes).reshape((rows, cols))
#     labels = [lookup[1] for lookup in bar_lookups]
    labels = None
#         "Dreamer",
    if ROT:
        pairs = pairs[::-1]
    for row in range(rows):
        col = 0
        ax = axes[row, col]
        bar_lookups = pairs[row]
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper)
        ax.set_xticklabels(["", ""])
        name = scenario_paper_names[bar_lookups[0][0]]
        name = name + "\n(empty)" if bar_lookups[0][3] == "easiest" else name
        name = name + "\n(sparse)" if bar_lookups[0][0] == "cathedral" and bar_lookups[0][3] == "easy" else name
        ax.set_ylabel(name)
        ax.set_ylim([0, 1.05])
        if ROT:
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
    ax.set_xticklabels(["Domain Specific", "SCR"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()

    # single hand picked plot
    fig, axes = plt.subplots(1, 1, num="test")
    N = 100
    bar_lookups = [
        ("alternate", "N3D", any_, "hardest", "navrep3daltenv", N, "SCR", "GPT", any_),
    ]
    ax = axes
    to_bar_chart(bar_lookups, ax)
    plt.show()

    # single plot with dreamer vs n3d
    fig, axes = plt.subplots(1, 1, num="tests")
    N = 100
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
    ax = axes
    to_bar_chart(bar_lookups, ax)

    plt.show()

    # xtest in gallery, cathedral
    fig, axes = plt.subplots(2, 1, num="xtests")
    N = 100
    bar_lookups = [
        ("gallery", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("gallery", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
        ("gallery", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ("gallery", "E2E", any_, "easy", "navrep3daltenv", N, any_, any_, any_),
        ("gallery", "E2E", any_, "easy", "navrep3dSCenv", N, any_, any_, any_),
        ("gallery", "E2E", any_, "easy", "navrep3dSCRenv", N, any_, any_, any_),
    ]
    ax = axes[0]
    to_bar_chart(bar_lookups)
    plt.show()
    N = 100
    bar_lookups = [
        ("cathedral", "N3D", any_, "easy", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("cathedral", "N3D", any_, "easy", "navrep3dSCenv", N, "SCR", "GPT", any_),
        ("cathedral", "N3D", any_, "easy", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ("cathedral", "E2E", any_, "easy", "navrep3daltenv", N, any_, any_, any_),
        ("cathedral", "E2E", any_, "easy", "navrep3dSCenv", N, any_, any_, any_),
        ("cathedral", "E2E", any_, "easy", "navrep3dSCRenv", N, any_, any_, any_),
    ]
    ax = axes[1]
    ax.to_bar_chart(bar_lookups, ax)
    plt.show()


if __name__ == "__main__":
    StrictFire(main)
