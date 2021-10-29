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

from navrep.scripts.plot_gym_training_progress import parse_logfiles

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
    "Random": "brown",
    "E2E": "grey",
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
        style = envname_styles[envname]
    return color, style

def get_variant(logpath):
    variant = None
    string = logpath.split("V64M64_")[-1]
    string = string.split(".")[0]
    # find variant in string
    for k in variant_colors:
        if string == k:
            variant = k
            break
    return variant

def get_envname(logpath):
    return os.path.basename(logpath).split("_")[0]

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


def plot_training_progress(logdirs, scenario=None, x_axis="total_steps", y_axis="reward"):
    logpaths, parents = parse_logfiles(logdirs)

    # get set of all scenarios in all logpaths
    all_scenarios = []
    for logpath in logpaths:
        S = pd.read_csv(logpath)
        scenarios = sorted(list(set(S["scenario"].values)))
        all_scenarios.extend(scenarios)
    all_scenarios = sorted(list(set(all_scenarios)))
    all_difficulties = ["all_difficulties"]

    if scenario is not None:
        all_scenarios = [scenario]

    if y_axis == "worst_perf":
        all_difficulties = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, "worst"]

    if x_axis == "wall_time":
        all_scenarios = ["all_scenarios"]
        all_difficulties = ["all_difficulties"]

    print()
    print("Plotting scenario rewards")
    print()
    plt.figure("scenario rewards")
    plt.clf()
    fig, axes = plt.subplots(len(all_difficulties), len(all_scenarios), num="scenario rewards")
    axes = np.array(axes).reshape((len(all_difficulties),len(all_scenarios))).T
    ax_key = {scenario: {diff: ax for diff, ax in zip(all_difficulties, ax_row)}
              for scenario, ax_row in zip(all_scenarios, axes)}

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parent", style="dim")
    table.add_column("Name")
    table.add_column("Steps", justify="right")
    table.add_column("Reward", justify="right")

    linegroups = []
    legends = []
    for logpath, parent in zip(logpaths, parents):
        linegroup = [] # regroup all lines from this log
        for scenario in all_scenarios:
            for difficulty in all_difficulties:
                ax = ax_key[scenario][difficulty]
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
                if difficulty == "all_difficulties":
                    scenario_S = scenario_S
                else:
                    if difficulty == "worst":
                        # will be used to compute worst of all
                        all_perfs = [scenario_S["goal_reached"].values.astype(np.float32) for _ in all_difficulties[:-1]]
                        for i in range(len(all_perfs)):
                            all_perfs[i][scenario_S["num_walls"] != all_difficulties[i]] = np.nan
                    else:
                        scenario_S = scenario_S[scenario_S["num_walls"] == difficulty]
                if len(scenario_S.values) == 0:
                    continue
                # million steps
                n = np.max(scenario_S["total_steps"].values) / MILLION
                # x axis
                if x_axis == "total_steps":
                    x = scenario_S["total_steps"].values
                    x = x / MILLION
                    xlabel = "Million Train Steps"
                elif x_axis == "wall_time":
                    try:
                        x = scenario_S["wall_time"].values
                    except KeyError:
                        print("{} has no wall_time info".format(logpath))
                        continue
                    x = md.epoch2num(x)
                    xlabel = "Wall Time"
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
                    rewards[0] = 0 # better would be to add a point at 0, 0, so the initial assumption is failure
                    if difficulty == "worst":
                        for perf in all_perfs:
                            perf[0] = 0
                        rewards = np.nanmin([smooth(perf, 0.99) for perf in all_perfs], axis=0)
                else:
                    raise NotImplementedError
                y = rewards
                smooth_y = smooth(y, 0.99)
                # plot main reward line
                line, = ax.plot(x, smooth_y, linewidth=1, linestyle=style, color=color)
                color = line.get_c()
                # add episode reward scatter
                scatter, = ax.plot(x, y, color=color, marker=',', linewidth=0, label=scenario)
                # add vertical line at end of finished runs
                if x_axis == "wall_time":
                    if n > N_:
                        ax.axvline(x[-1], linestyle=style, linewidth=1, color=color)
                    else:
                        ax.scatter(x[-1], y[-1], marker='>', facecolor="none", edgecolor=color)
                ax.set_ylim(ylim)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                ax.set_title("{} - {}".format(scenario, difficulty))
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
        if linegroup:
            linegroups.append(linegroup)
            legends.append(parent + ": " + logname)

    for scenario in all_scenarios:
        for difficulty in all_difficulties:
            ax = ax_key[scenario][difficulty]
            # add current time
            for ax in axes.reshape((-1,)):
                if x_axis == "wall_time":
                    ax.axvline(md.epoch2num(time.time()), color='k', linewidth=1)

    L = fig.legend([lines[0] for lines in linegroups], legends, bbox_to_anchor=(1.05, 1.))
    make_legend_pickable(L, linegroups)

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

def main(logdir="~/navrep3d",
         x_axis: str_enum(["wall_time", "total_steps"]) = "wall_time", # noqa
         y_axis: str_enum(["reward", "difficulty", "progress", "worst_perf"]) = "difficulty", # noqa (flake8 bug?)
         refresh: bool = typer.Option(False, help="Updates the plot every minute.")):
    logdirs = [os.path.expanduser(logdir),]
    print(x_axis.value)
    if refresh:
        while True:
            plt.ion()
            plot_training_progress(logdirs, x_axis=x_axis.value, y_axis=y_axis.value)
            plt.pause(60)
    else:
        plot_training_progress(logdirs, x_axis=x_axis.value, y_axis=y_axis.value)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
