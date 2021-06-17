import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as md
import matplotlib
from rich.console import Console
from rich.table import Table

from navrep.tools.commonargs import parse_plotting_args
from navrep.scripts.plot_gym_training_progress import parse_logfiles, smooth

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

MILLION = 1000000
N_ = 5 # millions of train steps before training is 'complete'

def get_color_and_style(logpath):
    return None, None

def plot_training_progress(logdirs, scenario="navrep3dtrain", x_axis="total_steps", y_axis="reward"):
    logpaths, parents = parse_logfiles(logdirs)

    # get set of all scenarios in all logpaths
    all_scenarios = []
    for logpath in logpaths:
        S = pd.read_csv(logpath)
        scenarios = sorted(list(set(S["scenario"].values)))
        all_scenarios.extend(scenarios)
    all_scenarios = sorted(list(set(all_scenarios)))

    print()
    print("Plotting scenario rewards")
    print()
    plt.ion()
    plt.figure("scenario rewards")
    plt.clf()
    fig, ax = plt.subplots(1, 1, num="scenario rewards")

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parent", style="dim")
    table.add_column("Name")
    table.add_column("Steps", justify="right")
    table.add_column("Reward", justify="right")

    lines = []
    legends = []
    for logpath, parent in zip(logpaths, parents):
        logname = os.path.basename(logpath)
        line = None
        color, style = get_color_and_style(logpath)
        S = pd.read_csv(logpath)
        scenario_S = S[S["scenario"] == scenario]
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
        if len(x) == 0:
            continue
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
        else:
            raise NotImplementedError
        y = rewards
        smooth_y = smooth(y, 0.99)
        # plot main reward line
        line, = ax.plot(x, smooth_y, linewidth=1, color=color)
        color = line.get_c()
        # add episode reward scatter
        ax.plot(x, y, color=color, marker=',', linewidth=0, label=scenario)
        # add vertical line at end of finished runs
        if x_axis == "wall_time":
            if n > N_:
                ax.axvline(x[-1], linestyle=style, linewidth=1, color=color)
            else:
                ax.scatter(x[-1], y[-1], marker='>', facecolor="none", edgecolor=color)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(scenario)
        if x_axis == "wall_time":
            xfmt = md.DateFormatter('%d-%b-%Y %H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.grid(which='minor', axis='x', linestyle='-')
            ax.grid(which='major', axis='x', linestyle='-')
        if line is not None:
            lines.append(line)
            legends.append(parent + ": " + logname)
    # add lucia best hline
    line = ax.axhline(85., color="red", linewidth=1, alpha=0.5)
    lines.append(line)
    legends.append("soadrl")
    # add current time
    if x_axis == "wall_time":
        ax.axvline(md.epoch2num(time.time()), color='k', linewidth=1)
    fig.legend(lines, legends, bbox_to_anchor=(1.05, 1.))
    console.print(table)


if __name__ == "__main__":
    args, _ = parse_plotting_args()

    if args.x_axis is None:
        args.x_axis = "total_steps"
    if args.y_axis is None:
        args.y_axis = "reward"

    while True:
        logdir = args.logdir
        if logdir is None:
            logdir = "~/navrep3d"
        logdirs = [os.path.expanduser(logdir),]
        plot_training_progress(logdirs, x_axis=args.x_axis, y_axis=args.y_axis)
        plt.pause(60)

