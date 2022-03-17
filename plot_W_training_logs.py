import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from strictfire import StrictFire

from plot_gym_training_progress import make_legend_pickable

def main(
    logdir="~/navdreams_data/wm_experiments",
    refresh=False,
    y_axis="lidar_test_error",
):
    logdir = os.path.expanduser(logdir)
    logdir = os.path.join(logdir, "logs/W")

    plt.close('all')
    while True:
        fig, ax = plt.subplots(1, 1, num="training log")
        plt.clf()
        logs = sorted(os.listdir(logdir))
        legends = []
        linegroups = []
        for log in logs:
            lines = []
            if not log.endswith(".csv"):
                continue
            path = os.path.join(logdir, log)
            data = pd.read_csv(path)
            x = data["step"].values
            y = data[y_axis].values
            y_valid_mask = np.logical_not(np.isnan(y))
            y = y[y_valid_mask]
            x = x[y_valid_mask]
            plt.ylabel(y_axis)
            line, = plt.plot(x, y, label=log)
            plt.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            lines.append(line)
            linegroups.append(lines)
            legends.append(log)
        L = fig.legend([lines[0] for lines in linegroups], legends)
        make_legend_pickable(L, linegroups)
        if refresh:
            plt.ion()
            plt.pause(10.)
        else:
            plt.show()
            break


if __name__ == "__main__":
    StrictFire(main)

