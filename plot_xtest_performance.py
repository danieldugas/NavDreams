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
    "kozehd": "replica",
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
    logdir="~/navdreams_data/results/test/",
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

    def to_bar_chart(bar_lookups, ax, labels=None, hide_error=False, merge_seeds=False):
        if len(bar_lookups) == 0:
            return
        reachedgoals = []
        asy_errors = []
        crashes = []
        crashesother = []
        timeouts = []
        foundkeys = []
        spoilts = []
        for lookup in bar_lookups:
            matches, keys = find_matches_in_data(lookup, data, alert_if_not_found=True)
            if len(matches) != 1 and not merge_seeds:
                raise ValueError("Matches != 1:\nfor\n{}\nfound\n{}".format(lookup, keys))
            seeds_successes = []
            seeds_timeouts = []
            seeds_crashes = []
            seeds_crashesother = []
            seeds_spread = []
            for arrays, key in zip(matches, keys):
                # successes
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
                splits = np.array_split(successes, 2)
                splits = [np.mean(s) for s in splits]
                spread = max(splits) - min(splits)
                seeds_successes.append(np.mean(successes))
                seeds_timeouts.append(np.mean(causes == "Timeout"))
                seeds_crashes.append(np.mean(causes == "Collision"))
                seeds_crashesother.append(np.mean(causes == "Collision from other agent"))
                seeds_spread.append(spread)

            # merge seeds into one bar
            tol = 0.15
            seeds_spread = np.array(seeds_spread)
            spoilt = np.all(seeds_spread > tol)
            if spoilt:  # still show something
                spoilt = np.mean(seeds_spread) / 2
            else:
                seeds_successes = np.array(seeds_successes)[seeds_spread <= tol]
                seeds_timeouts = np.array(seeds_timeouts)[seeds_spread <= tol]
                seeds_crashes = np.array(seeds_crashes)[seeds_spread <= tol]
                seeds_crashesother = np.array(seeds_crashesother)[seeds_spread <= tol]
            # add single bar to bars
            spoilts.append(spoilt)
            timeouts.append(np.mean(seeds_timeouts))
            crashes.append(np.mean(seeds_crashes))
            crashesother.append(np.mean(seeds_crashesother))
            reachedgoals.append(np.mean(seeds_successes))
            asy_error = [abs(min(seeds_successes)-np.mean(seeds_successes)),
                         abs(max(seeds_successes)-np.mean(seeds_successes))]
            asy_errors.append(asy_error)
            # label
            build, mtype, ckpt, difficulty, trainenv, n_episodes, wmscope, wmtype, uid = lookup
            foundkeys.append(key)
        if labels is None:
            labels = [str(k) for k in foundkeys]

        spoilts = np.array(spoilts)
        reachedgoals = np.array(reachedgoals)
        asy_errors = np.array(asy_errors).reshape((len(reachedgoals), 2)).T
        timeouts = np.array(timeouts)
        crashes = np.array(crashes)
        crashesother = np.array(crashesother)
        labels = np.array(labels)
        if hide_error:
            asy_errors = None
        ax.bar(labels, reachedgoals, yerr=asy_errors, color="mediumseagreen")
        ax.bar(labels, timeouts, bottom=reachedgoals, color="lightgrey")
        ax.bar(labels, crashes, bottom=reachedgoals+timeouts, color="orange")
        ax.bar(labels, crashesother, bottom=reachedgoals+timeouts+crashes, color="tomato")
        plt.setp(ax.get_xticklabels(), Fontsize=12)
        if not hide_error:
            ax.bar(labels[spoilts > 0], (reachedgoals - spoilts)[spoilts > 0], color="blue")

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

    # legend
    if False:
        plt.figure()
        plt.bar([0], [1], label="success", color="mediumseagreen")
        plt.bar([0], [1], label="timeout", color="lightgrey")
        plt.bar([0], [1], label="object collision", color="orange")
        plt.bar([0], [1], label="person collision", color="tomato")
        plt.legend(ncol=4)
        plt.show()

    # single plot with best in each
    ROT = True
    N = 100
    pairs = [
        [
            ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", any_), # noqa
            ("alternate", "E2E", any_, "hardest", "navrep3daltenv", N, any_, any_, any_), # noqa
        ], [
            ("city", "N3D", "bestckpt", "hardest", "navrep3dcityenv", N, "SCR", "GPT", any_), # noqa
            ("city", "E2E", any_, "hardest", "navrep3dcityenv", N, any_, any_, any_), # noqa
        ], [
            ("office", "N3D", "bestckpt", "random", "navrep3dofficeenv", N, "SCR", "GPT", any_), # noqa
            ("office", "E2E", any_, "random", "navrep3dofficeenv", N, any_, any_, any_), # noqa
        ], [
            ("staticasl", "N3D", "bestckpt", "medium", "navrep3daslfixedenv", N, "SCR", "GPT", any_), # noqa
            ("staticasl", "E2E", any_, "medium", "navrep3daslfixedenv", N, any_, any_, any_), # noqa
        ], [
            ("cathedral", "N3D", "bestckpt", "medium", "navrep3dcathedralenv", N, "SCR", "GPT", any_), # noqa
            ("cathedral", "E2E", any_, "medium", "navrep3dcathedralenv", N, any_, any_, any_), # noqa
        ], [
            ("gallery", "N3D", "bestckpt", "easy", "navrep3dgalleryenv", N, "SCR", "GPT", any_), # noqa
            ("gallery", "E2E", any_, "easy", "navrep3dgalleryenv", N, any_, any_, any_), # noqa
#         ], [
#             ("kozehd", "N3D", "bestckpt", "easier", "navrep3dkozehdrsenv", N, "K2", "GPT", any_), # noqa
#             ("kozehd", "E2E", any_, "easier", "navrep3dkozehdrsenv", N, any_, any_, any_), # noqa
#         ], [
#             ("kozehd", "N3D", "bestckpt", "easy", "navrep3dkozehdrsenv", N, "K2", "GPT", any_), # noqa
#             ("kozehd", "E2E", any_, "easy", "navrep3dkozehdrsenv", N, any_, any_, any_), # noqa
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
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper, merge_seeds=True)
        ax.set_xticklabels(["", ""])
        name = scenario_paper_names[bar_lookups[0][0]]
        name = name + "\n(empty)" if bar_lookups[0][3] == "easiest" else name
        ax.set_ylabel(name, fontsize=12)
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
        ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", any_),
        ("alternate", "E2E", any_, "hardest", "navrep3daltenv", N, any_, any_, any_),
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
    to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper, merge_seeds=True)
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
            ("kozehd", "N3D", any_, "easier", "navrep3daltenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easier", "navrep3dSCenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easier", "navrep3dSCRenv", N, "SCR", "GPT", any_),
            ("kozehd", "N3D", any_, "easier", "navrep3dkozehdrsenv", N, "K2", "GPT", any_),
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
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper, merge_seeds=True)
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
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper, merge_seeds=True)
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
            ("alternate", "N3D", "bestckpt", "hardest", "navrep3daltenv", N, "SCR", "GPT", any_), # noqa
            ("alternate", "N3D", any_, "hardest", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [
            ("city", "N3D", "bestckpt", "hardest", "navrep3dcityenv", N, "SCR", "GPT", any_), # noqa
            ("city", "N3D", any_, "hardest", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [
            ("office", "N3D", "bestckpt", "random", "navrep3dofficeenv", N, "SCR", "GPT", any_), # noqa
            ("office", "N3D", any_, "random", "navrep3dSCRenv", N, "SCR", "GPT", any_),
        ], [ # this one is wrong! training in old but testing in fixed env
            ("staticasl", "N3D", "bestckpt", "medium", "navrep3daslfixedenv", N, "SCR", "GPT", any_), # noqa
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
        to_bar_chart(bar_lookups, ax, labels=labels, hide_error=paper, merge_seeds=True)
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
