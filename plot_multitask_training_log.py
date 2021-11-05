import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import typer

def main(refresh : bool = True, logdir : str = None, paper : bool = False):
    if logdir is None:
        logdir = os.path.expanduser("~/navrep3d/logs/multitask")

    plt.close('all')
    while True:
        plt.figure("training log")
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num="training log")
        logs = sorted(os.listdir(logdir))
        legend = []
        lines = []
        for log in logs:
            if not log.endswith(".csv"):
                continue
            if "segmenter" in log:
                ax = ax1
                ax.set_title("segmentation error")
                ax.set_ylabel("segmentation BCE error")
            elif "depth" in log:
                ax = ax2
                ax.set_title("depth error")
                ax.set_ylabel("depth MSE [hm^2]")
            else:
                raise NotImplementedError
            path = os.path.join(logdir, log)
            data = pd.read_csv(path)
            x = data["step"].values
            y = data["test_error"].values
            if paper:
                if x[-1] < 80000:
                    continue
                if "sequence" in log:
                    continue
            style = "solid"
            color = None
            if paper:
                if "baseline" in log:
                    style = "dashed"
                    color = 'k'
            line, = ax.plot(x, y, linestyle=style, color=color, label=log)
            ax.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            ax.set_xlabel("training steps")
            lines.append(line)
            legend.append(log)
        ax1.axhline(0, linewidth=1, color='k')
        ax2.axhline(0, linewidth=1, color='k')
        if paper:
            ax1.set_ylim((0, 0.05))
            ax2.set_ylim((0, 0.005))
        ax1.legend()   # quick-search : plot tcn training logs
        if refresh:
            plt.ion()
            plt.pause(10.)
        else:
            plt.show()
            break


if __name__ == "__main__":
    typer.run(main)
