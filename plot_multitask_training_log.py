import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import typer

def main(refresh : bool = True, logdir : str = None):
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
            legend.append(log)
            data = pd.read_csv(path)
            x = data["step"].values
            y = data["epoch_loss"].values
            line, = ax.plot(x, y, label=log)
            ax.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            ax.set_xlabel("training steps")
            lines.append(line)
        plt.legend()   # quick-search : plot tcn training logs
        if refresh:
            plt.ion()
            plt.pause(10.)
        else:
            plt.show()
            break


if __name__ == "__main__":
    typer.run(main)
