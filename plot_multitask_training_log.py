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
            # name
            name = log
            if paper:
                if "baseline" in log:
                    name = "Task-specific features"
                elif "E2E" in log:
                    name = "End-to-end features"
                elif "N3D" in log:
                    name = "NavRep3D features"
                else:
                    raise NotImplementedError
            # title and axes
            if "segmenter" in log:
                ax = ax1
                ax.set_title("Segmentation Error")
                ax.set_ylabel("Segmentation BCE Error")
            elif "depth" in log:
                ax = ax2
                ax.set_title("Depth Error")
                ax.set_ylabel("Depth Mean Square Proportional Error")
            else:
                raise NotImplementedError
            # read data
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
            line, = ax.plot(x, y, linestyle=style, color=color, label=name)
            ax.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            ax.set_xlabel("training steps")
            lines.append(line)
            legend.append(name)
        ax1.axhline(0, linewidth=1, color='k')
        ax2.axhline(0, linewidth=1, color='k')
        if paper:
            ax1.set_ylim((0, 0.05))
            ax2.set_ylim((0, 0.1))
            ax1.legend()
        else:
            ax1.legend()
            ax2.legend()
        if refresh:
            plt.ion()
            plt.pause(10.)
        else:
            plt.show()
            break


if __name__ == "__main__":
    typer.run(main)
