import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as md
import matplotlib
from rich.console import Console
from rich.table import Table
from enum import Enum
import typer
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

MILLION = 1000000
N_ = 5 # millions of train steps before training is 'complete'

envname_styles = {
    "navrep3daltencodedenv": "dashed",
    "e2enavrep3dtrain": "solid",
    "navrep3dtrainencodedenv": "solid",
    "navrep3daltenv": "dashed",
    "navrep3dtrainenv": "solid",
}
variant_colors = {
    "S": "lightskyblue",
    "Salt": "mediumseagreen",
    "SC": "khaki",
    "SCR": "red",
    "R": "mediumorchid",
    "Random": "brown",
    "E2E": "grey",
    "DREAMER": "orange",
    "SCRK": "blue",
    "SCRK2": "violet",
    "K2": "red",
}
clean_scenario_names = {
    "navrep3dalt": "simple",
    "navrep3dcity": "city",
    "navrep3doffice": "office",
    "navrep3dasl": "modern",
    "navrep3dcathedral": "cathedral",
    "navrep3dgallery": "gallery",
    "navrep3dkozehd": "replica",
    "all_scenarios": "",
}

def smooth(x, weight):
    """ Weight between 0 and 1 """
    last = x[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in x:
        if np.isnan(point):
            smoothed.append(last)
            continue
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def color_and_style(variant, envname):
    color = None
    style = None
    if variant is not None:
        color = variant_colors[variant]
    if envname is not None:
        if envname in envname_styles:
            style = envname_styles[envname]
    return color, style

def get_variant(logpath):
    if "DREAMER" in logpath:
        return "DREAMER"
    if "E2E" in logpath:
        return "E2E"
    variant = None
    string = logpath.split("V64M64_")[-1]
    string = string.split(".")[0]
    string = string.split("_")[0]
    # find variant in string
    for k in variant_colors:
        if string == k:
            variant = k
            break
    return variant

def get_envname(logpath):
    envname = os.path.basename(logpath).split("_")[0]
    return envname.replace("encoded", "")

def set_visible(lines, visible):
    if isinstance(lines, list):
        for line in lines:
            line.set_visible(visible)
    else:
        lines.set_visible(visible)

def get_visible(lines):
    if isinstance(lines, list):
        for line in lines:
            return line.get_visible()
        return False
    else:
        return lines.get_visible()

def parse_logfiles(navrep_dirs, logfolder=None, exclude=None, include=None):
    """
    navrep_dirs : ["~/navdreams_data/results", "~/navdreams_data/wm_experiments"]
    logfolder : "logs/gym"
    """
    logfolder = "logs/gym" if logfolder is None else logfolder
    best_navrep_names = [os.path.basename(path) for path in navrep_dirs]

    all_logpaths = []
    all_parents = []
    for name, dir_ in zip(best_navrep_names, navrep_dirs):
        logdir = os.path.join(dir_, logfolder)
        try:
            logfiles = sorted([file for file in os.listdir(logdir) if ".csv" in file])
        except FileNotFoundError:
            logfiles = []
        if exclude is not None:
            logfiles = [file for file in logfiles if exclude not in file]
        if include is not None:
            logfiles = [file for file in logfiles if include in file]
        logpaths = [os.path.join(logdir, logfile) for logfile in logfiles]
        logparents = [name for _ in logfiles]
        all_logpaths.extend(logpaths)
        all_parents.extend(logparents)
    return all_logpaths, all_parents

def plot_training_progress(logdirs, scenario=None, x_axis="total_steps", y_axis="reward",
                           no_dots=False,
                           exclude=None, include=None,
                           environment=None,
                           finetune=False, smoothness=None):
    if smoothness is None:
        smoothness = 0.999
    logfolder = "logs/finetune" if finetune else None
    logpaths, parents = parse_logfiles(logdirs, logfolder=logfolder, exclude=exclude, include=include)

    # get set of all scenarios in all logpaths
    all_scenarios = []
    all_environments = []
    for logpath in logpaths:
        S = pd.read_csv(logpath)
        scenarios = sorted(list(set(S["scenario"].values)))
        all_scenarios.extend(scenarios)
        envname = get_envname(logpath)
        all_environments.append(envname)
    all_scenarios = sorted(list(set(all_scenarios)))
    all_environments = sorted(list(set(all_environments)))
    def custom_sort(to_sort, sortlist):
        to_sort = sorted(to_sort)
        for scenario in sortlist[::-1]:
            if scenario in to_sort:
                to_sort.insert(0, to_sort.pop(to_sort.index(scenario)))
        return to_sort
    sortlist = ["navrep3dtrain", "navrep3dalt", "navrep3dcity", "navrep3doffice", "navrep3dasl"]
    all_scenarios = custom_sort(all_scenarios, sortlist)
    sortlist = ["navrep3dtrainenv", "navrep3daltenv", "navrep3dSCenv", "navrep3dSCRenv", "navrep3dstaticaslenv"] # noqa
    all_environments = custom_sort(all_environments, sortlist)
    all_difficulties = ["all_difficulties"]
    rows_are_environments = True

    if scenario is not None:
        all_scenarios = [scenario]

    if environment is not None:
        if environment not in all_environments:
            raise ValueError("{} not found in {}".format(environment, all_environments))
        all_environments = [environment]

    if y_axis == "worst_perf":
        all_difficulties = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, "worst"]
        all_environments = ["all_environments"] if len(all_environments) > 1 else all_environments
        rows_are_environments = False

    if x_axis == "wall_time":
        all_scenarios = ["all_scenarios"]
        all_difficulties = ["all_difficulties"]
        all_environments = ["all_environments"]

    print()
    print("Plotting scenario rewards")
    print()
    plt.figure("scenario rewards")
    plt.clf()
    row_names = all_environments if rows_are_environments else all_difficulties
    fig, axes = plt.subplots(len(row_names), len(all_scenarios), num="scenario rewards")
    axes = np.array(axes).reshape((len(row_names),len(all_scenarios))).T
    ax_key = {scenario: {name: ax for name, ax in zip(row_names, ax_row)}
              for scenario, ax_row in zip(all_scenarios, axes)}

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parent", style="dim")
    table.add_column("Name")
    table.add_column("Steps", justify="right")
    table.add_column("Reward", justify="right")

    linegroups = []
    legends = []
    for logpath, parent in tqdm(zip(logpaths, parents), total=len(logpaths)):
        linegroup = [] # regroup all lines from this log
        for scenario in all_scenarios:
            for row_name in row_names:
                ax = ax_key[scenario][row_name]
                logname = os.path.basename(logpath)
                line = None
                variant = get_variant(logpath)
                envname = get_envname(logpath)
                color, _ = color_and_style(variant, envname)
                S = pd.read_csv(logpath)
                if scenario == "all_scenarios":
                    _, style = color_and_style(variant, envname)
                    scenario_S = S
                else:
                    style = None
                    scenario_S = S[S["scenario"] == scenario]
                if rows_are_environments:
                    ax_environment = row_name
                    if ax_environment != "all_environments":
                        if envname != ax_environment:
                            continue
                else:
                    ax_difficulty = row_name
                    if ax_difficulty == "all_difficulties":
                        scenario_S = scenario_S
                    else:
                        if ax_difficulty == "worst":
                            # will be used to compute worst of all
                            all_perfs = [scenario_S["goal_reached"].values.astype(np.float32)
                                         for _ in all_difficulties[:-1]]
                            for i in range(len(all_perfs)):
                                all_perfs[i][scenario_S["num_walls"] != all_difficulties[i]] = np.nan
                        else:
                            scenario_S = scenario_S[scenario_S["num_walls"] == ax_difficulty]
                if len(scenario_S.values) == 0:
                    continue
                # million steps
                n = np.max(scenario_S["total_steps"].values) / MILLION
                # x axis
                if x_axis == "total_steps":
                    x = scenario_S["total_steps"].values
                    x = x / MILLION
                    xlabel = "Million Train Steps"
                    ax.set_xlim([0, 5])
                elif x_axis == "wall_time":
                    try:
                        x = scenario_S["wall_time"].values
                    except KeyError:
                        print("{} has no wall_time info".format(logpath))
                        continue
                    x = md.epoch2num(x)
                    xlabel = "Wall Time"
                elif x_axis == "train_time":
                    try:
                        x = scenario_S["wall_time"].values
                        x = x - x[0]
                    except KeyError:
                        print("{} has no wall_time info".format(logpath))
                        continue
                    xlabel = "Train Time"
                else:
                    raise NotImplementedError
                # y axis
                if y_axis == "reward":
                    ylabel = "reward"
                    ylim = [-26, 101]
                    rewards = scenario_S["reward"].values
                elif y_axis == "difficulty":
                    ylabel = "num_walls"
                    ylim = [-1, 22]
                    rewards = scenario_S["num_walls"].values
                elif y_axis == "progress":
                    ylabel = "total steps [million]"
                    ylim = [-1, 10]
                    rewards = scenario_S["total_steps"].values / MILLION
                elif y_axis == "worst_perf":
                    ylabel = "success rate [0-1]"
                    ylim = [-0.1, 1.1]
                    rewards = scenario_S["goal_reached"].values
                    rewards[0] = 0 # better would be to add a point at 0, 0, so the init. assumpt. is failure
                    if ax_difficulty == "worst":
                        for perf in all_perfs:
                            perf[0] = 0
                        rewards = np.nanmin([smooth(perf, smoothness) for perf in all_perfs], axis=0)
                else:
                    raise NotImplementedError
                y = rewards
                smooth_y = smooth(y, smoothness if not variant == "DREAMER" else 0.995)
                # plot main reward line
                line, = ax.plot(x, smooth_y, linewidth=1, linestyle=style, color=color)
                color = line.get_c()
                # add episode reward scatter
                scatter = None
                if not no_dots:
                    scatter, = ax.plot(x, y, color=color, marker=',', linewidth=0, label=scenario)
                top = ax.scatter(x[np.argmax(smooth_y)], np.max(smooth_y), marker='o', facecolor="none",
                                 edgecolor=color)

                # add vertical line at end of finished runs
                if x_axis == "wall_time":
                    if n > N_:
                        ax.axvline(x[-1], linestyle=style, linewidth=1, color=color)
                    else:
                        ax.scatter(x[-1], y[-1], marker='>', facecolor="none", edgecolor=color)
                ax.set_ylim(ylim)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                ax.set_title("{} - {}".format(scenario, row_name))
                if x_axis == "wall_time":
                    xfmt = md.DateFormatter('%d-%b-%Y %H:%M:%S')
                    ax.xaxis.set_major_formatter(xfmt)
                    from matplotlib.ticker import MultipleLocator
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
                    ax.xaxis.set_major_locator(MultipleLocator(3))
                    ax.grid(which='minor', axis='x', linestyle='-')
                    ax.grid(which='major', axis='x', linestyle='-')
                if line is not None:
                    linegroup.append(line)
                if scatter is not None:
                    linegroup.append(scatter)
                if top is not None:
                    linegroup.append(top)
        if linegroup:
            linegroups.append(linegroup)
            legends.append(parent + ": " + logname)

    for scenario in all_scenarios:
        for row_name in row_names:
            ax = ax_key[scenario][row_name]
            # add current time
            for ax in axes.reshape((-1,)):
                if x_axis == "wall_time":
                    ax.axvline(md.epoch2num(time.time()), color='k', linewidth=1)

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

    console.print(table)

def plot_multiseed_performance(logpaths, parents, variant, scenario, envname, ax):
    x_axis = "total_steps"
    y_axis = "difficulty"
    smoothness = 0.999
    max_difficulty = 18.
    y_max = 1.0
    # calculate smoothed lines for each plot
    smoothed_curves = []
    for logpath, parent in zip(logpaths, parents):
        line = None
        if variant != get_variant(logpath):
            continue
        if envname != get_envname(logpath):
            continue
        color, _ = color_and_style(variant, envname)
        S = pd.read_csv(logpath)
        if scenario == "all_scenarios":
            logscenario = S["scenario"].unique()[0]
            scenario_S = S
        else:
            logscenario = scenario
            scenario_S = S[S["scenario"] == scenario]
        if len(scenario_S.values) == 0:
            continue
        # adjust difficulty
        if logscenario == "navrep3doffice":
            max_difficulty = 10.
        elif logscenario == "navrep3dasl":
            max_difficulty = 50.
            y_max = 0.4
        elif logscenario == "navrep3dcathedral":
            max_difficulty = 50.
            y_max = 0.5
        elif logscenario == "navrep3dgallery":
            max_difficulty = 50.
            y_max = 0.16
        elif logscenario == "navrep3dkozehd":
            max_difficulty = 20.
            y_max = 0.2
        # x axis
        if x_axis == "total_steps":
            x = scenario_S["total_steps"].values
            x = x / MILLION
            xlabel = "Million Train Steps"
        else:
            raise NotImplementedError
        # y axis
        if y_axis == "difficulty":
            ylabel = "average scenario difficulty"
            ylim = [-0., y_max]
            rewards = scenario_S["num_walls"].values / max_difficulty
        else:
            raise NotImplementedError
        y = rewards
        smooth_y = smooth(y, smoothness)
        smoothed_curves.append((x, smooth_y))

    # remove incomplete
    smoothed_curves = [(x, y) for x, y in smoothed_curves if max(x) > 4.5]

    # minmax curve
    n_seeds = len(smoothed_curves)
    if n_seeds == 0:
        return [], 0
    style = None
    if n_seeds < 3:
        style = "dotted"
    end = np.min([np.max(x) for x, y in smoothed_curves])
    common_x = np.arange(0, end, 10000. / MILLION)
    filled_smoothed_curves = [np.interp(common_x, x, y) for x, y in smoothed_curves]
    Y = np.array(filled_smoothed_curves)
    mean_ = np.mean(Y, axis=0)
    min_ = np.min(Y, axis=0)
    max_ = np.max(Y, axis=0)

    linegroup = [] # regroup all lines from this variant
    ax.set_ylim(ylim)
    ax.set_xlim([0, 5])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title("{}".format(clean_scenario_names[logscenario]))
    line, = ax.plot(common_x, mean_, linewidth=1, linestyle=style, color=color)
    color = line.get_c()
    area = ax.fill_between(common_x, min_, max_, color=color, alpha=0.1)
    linegroup.append(line)
    linegroup.append(area)
    return linegroup, n_seeds

def plot_training_results(logdirs, envname="navrep3daltenv", logfolder=None):
    logpaths, parents = parse_logfiles(logdirs, logfolder=logfolder)

    if envname == "navrep3daltenv":
        scenario = "navrep3dalt"
    elif envname == "navrep3daslenv":
        scenario = "navrep3dasl"
    else:
        raise NotImplementedError
    all_variants = ["R", "SCR", "SC", "Salt", "E2E"]

    fig, ax = plt.subplots(1, 1, num=envname + "training results")

    linegroups = []
    legends = []
    for variant in all_variants:
        linegroup, n = plot_multiseed_performance(logpaths, parents, variant, scenario, envname, ax)
        if linegroup:
            linegroups.append(linegroup)
            legends.append(variant)

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

def plot_direct_training_results(logdirs,
                                 envnames=["navrep3daltenv",
                                           "navrep3dcityenv",
                                           "navrep3dofficeenv",
                                           "navrep3daslenv",
                                           "navrep3dcathedralenv",
                                           "navrep3dgalleryenv",
#                                            "navrep3dkozehdrsenv",
                                           ],
                                 logfolder=None):
    logpaths, parents = parse_logfiles(logdirs, logfolder=logfolder)

    scenarios = ["all_scenarios"]
    all_variants = ["SCR", "E2E", "K2"]

    fig, axes = plt.subplots(len(scenarios), len(envnames), num="_".join(envnames) + "x training results")
    axes = np.array(axes).reshape((len(scenarios), len(envnames)))

    linegroups = []
    legends = []
    seeds_count = {}
    for variant in tqdm(all_variants):
        variant_linegroup = []
        for scenario, ax_row in zip(scenarios, axes):
            for envname, ax in zip(envnames, ax_row):
                n_seeds = 0
                seeds_count[(variant, envname)] = n_seeds
                if envname == "navrep3dkozehdrsenv" and variant == "SCR":
                    continue
                linegroup, n = plot_multiseed_performance(logpaths, parents, variant, scenario, envname, ax)
                if linegroup:
                    variant_linegroup.extend(linegroup)
                n_seeds = max(n_seeds, n)
                seeds_count[(variant, envname)] = n_seeds
                ax.set_ylabel("")
                ax.set_xlabel("")
        if variant_linegroup:
            linegroups.append(variant_linegroup)
            legends.append("End-to-End" if variant == "E2E" else "WorldModel")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode Difficulty\n(Rolling Average)")
    for ax in axes[0, 3:4]:
        ax.set_xlabel("Million Training Steps")
    for ax in axes[0, :]:
        for tick in ax.get_yticklabels():
            tick.set_rotation(90)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("")
    for variant in all_variants:
        table.add_column(variant)
    for envname in envnames:
        table.add_row(envname, *[str(seeds_count[(variant, envname)]) for variant in all_variants])
    console.print(table)

def plot_simple_xtraining_results(
    logdirs,
    envnames=["navrep3daltenv", "navrep3dSCRenv", "navrep3dSCenv", "navrep3daslenv"],
    logfolder=None
):
    logpaths, parents = parse_logfiles(logdirs, logfolder=logfolder)

    scenarios = ["all_scenarios"]
    all_variants = ["R", "SCR", "SC", "Salt", "E2E"]

    fig, axes = plt.subplots(len(scenarios), len(envnames), num="_".join(envnames) + "x training results")
    axes = np.array(axes).reshape((len(scenarios), len(envnames)))

    linegroups = []
    legends = []
    seeds_count = {}
    for variant in tqdm(all_variants):
        variant_linegroup = []
        for _, ax_row in zip(scenarios, axes):
            for envname, ax in zip(envnames, ax_row):
                if envname == "navrep3daltenv":
                    scenario = "navrep3dalt"
                elif envname == "navrep3dSCenv":
                    scenario = "navrep3doffice"
                elif envname == "navrep3dSCRenv":
                    scenario = "navrep3dcity"
                elif envname == "navrep3daslenv":
                    scenario = "navrep3dasl"
                n_seeds = 0
                linegroup, n = plot_multiseed_performance(logpaths, parents, variant, scenario, envname, ax)
                ax.set_xlabel("")
                if linegroup:
                    variant_linegroup.extend(linegroup)
                n_seeds = max(n_seeds, n)
                seeds_count[(variant, envname)] = n_seeds
                ax.set_xlabel("")
                ax.set_ylabel("")
        if variant_linegroup:
            linegroups.append(variant_linegroup)
            legends.append(variant)
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode Difficulty\n(Rolling Average)")
    for ax in axes[0, :]:
        ax.set_xlabel("Million Training Steps")

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("")
    for variant in all_variants:
        table.add_column(variant)
    for envname in envnames:
        table.add_row(envname, *[str(seeds_count[(variant, envname)]) for variant in all_variants])
    console.print(table)

def plot_xtraining_results(logdirs, envnames=["navrep3dSCenv", "navrep3dSCRenv"], logfolder=None):
    logpaths, parents = parse_logfiles(logdirs, logfolder=logfolder)

    if "navrep3dSCRenv" in envnames:
        scenarios = ["navrep3dalt", "navrep3dcity", "navrep3doffice", "navrep3dasl"]
    elif "navrep3dSCenv" in envnames:
        scenarios = ["navrep3dalt", "navrep3dcity", "navrep3doffice"]
    all_variants = ["R", "SCR", "SC", "Salt", "E2E"]

    fig, axes = plt.subplots(len(scenarios), len(envnames), num="_".join(envnames) + "x training results")
    axes = axes.reshape((len(scenarios), len(envnames)))

    linegroups = []
    legends = []
    seeds_count = {}
    for variant in tqdm(all_variants):
        variant_linegroup = []
        for scenario, ax_row in zip(scenarios, axes):
            for envname, ax in zip(envnames, ax_row):
                n_seeds = 0
                linegroup, n = plot_multiseed_performance(logpaths, parents, variant, scenario, envname, ax)
                ax.set_xlabel("")
                if linegroup:
                    variant_linegroup.extend(linegroup)
                n_seeds = max(n_seeds, n)
                seeds_count[(variant, envname)] = n_seeds
        ax.set_xlabel("Million Training Steps")
        if variant_linegroup:
            linegroups.append(variant_linegroup)
            legends.append(variant)

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("")
    for variant in all_variants:
        table.add_column(variant)
    for envname in envnames:
        table.add_row(envname, *[str(seeds_count[(variant, envname)]) for variant in all_variants])
    console.print(table)

def make_legend_pickable(legend, lines):
    """ Allows clicking the legend to toggle line visibility
    arguments:
        legend: the legend object (output of plt.legend())
        lines: list of line objects corresponding to legend items.
               should be of same length as legend.get_lines()
               Note: line objects can be anything which has a set_visible(bool is_visible) method
    """
    lineobjects = {}
    legenditems = legend.get_lines()
    for item, line in zip(legenditems, lines):
        item.set_picker(True)
        item.set_pickradius(10)
        lineobjects[item] = line
    def on_click_legenditem(event):
        legenditem = event.artist
        is_visible = get_visible(legenditem)
        set_visible(lineobjects[legenditem], not is_visible)
        set_visible(legenditem, not is_visible)
        plt.gcf().canvas.draw()
    plt.connect('pick_event', on_click_legenditem)

def str_enum(options: list):
    return Enum("", {n: n for n in options}, type=str)

def main(logdir="~/navdreams_data/results",
         x_axis: str_enum(["wall_time", "train_time", "total_steps"]) = "wall_time", # noqa
         y_axis: str_enum(["reward", "difficulty", "progress", "worst_perf"]) = "difficulty", # noqa (flake8 bug?)
         no_dots: bool = False,
         refresh: bool = typer.Option(False, help="Updates the plot every minute."),
         finetune : bool = False,
         smoothness : float = None,
         paper : bool = False,
         env : str = None,
         scenario : str = None,
         exclude : str = None,
         include : str = None,
         ):
    logdirs = [os.path.expanduser(logdir),]
    print(x_axis.value)
    if paper:
        plot_direct_training_results(logdirs)
        plt.show()
        plot_simple_xtraining_results(logdirs)
        plt.show()
        plot_training_results(logdirs)
        plot_xtraining_results(logdirs, envnames=["navrep3dSCenv"])
        plot_xtraining_results(logdirs, envnames=["navrep3dSCRenv"])
        plot_training_results(logdirs, envname="navrep3daslenv")
        plt.show()
        return
    if refresh:
        while True:
            plt.ion()
            plot_training_progress(logdirs, x_axis=x_axis.value, y_axis=y_axis.value,
                                   no_dots=no_dots,
                                   exclude=exclude, include=include,
                                   finetune=finetune, smoothness=smoothness,
                                   environment=env, scenario=scenario)
            plt.pause(60)
    else:
        plot_training_progress(logdirs, x_axis=x_axis.value, y_axis=y_axis.value,
                               no_dots=no_dots,
                               exclude=exclude, include=include,
                               finetune=finetune, smoothness=smoothness,
                               environment=env, scenario=scenario)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
