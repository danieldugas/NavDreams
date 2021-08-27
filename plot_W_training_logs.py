import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from fire import Fire

def main(
    logdir="~/navrep3d_W",
):
    logdir = os.path.expanduser(logdir)
    logdir = os.path.join(logdir, "logs/W")

    plt.close('all')
    while True:
        plt.figure("training log")
        plt.clf()
        logs = sorted(os.listdir(logdir))
        legend = []
        lines = []
        for log in logs:
            if not log.endswith(".csv"):
                continue
            path = os.path.join(logdir, log)
            legend.append(log)
            data = pd.read_csv(path)
            x = data["step"].values
            y = data["cost"].values
            line, = plt.plot(x, y, label=log)
            plt.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            lines.append(line)
        plt.legend()   # quick-search : plot tcn training logs
        plt.ion()
        plt.pause(10.)


if __name__ == "__main__":
    Fire(main)

