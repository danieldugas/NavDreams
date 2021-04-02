import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as md
import matplotlib
from rich.console import Console
from rich.table import Table
from datetime import timedelta

from navrep.tools.commonargs import parse_plotting_args
from navrep.scripts.plot_gym_training_progress import plot_per_scenario_training_progress, parse_logfiles

SCENARIO_AXES = {
    "navrep3dtrain"         : (0, 0),
}
ERROR_IF_MISSING = True


if __name__ == "__main__":
    args, _ = parse_plotting_args()

    if args.scenario is not None:
        SCENARIO_AXES = {args.scenario : (0, 0)}  # show only 1 scenario
        ERROR_IF_MISSING = False
    if args.scenario == "navrep3dtrain":
        SCENARIO_AXES = {"navrep3dtrain" : (0, 0), "navrep3dval" : (1, 0)}
    if args.x_axis is None:
        args.x_axis = "train_steps"

    while True:
        logdir = args.logdir
        if logdir is None:
            logdir = "~/navrep3d"
        logdirs = [os.path.expanduser(logdir),]
        plot_per_scenario_training_progress(logdirs, SCENARIO_AXES, ERROR_IF_MISSING,
                                            x_axis=args.x_axis)
        plt.pause(60)

