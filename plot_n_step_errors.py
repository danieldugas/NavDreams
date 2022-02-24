from matplotlib import pyplot as plt
import numpy as np
import os
from strictfire import StrictFire

from plot_gym_training_progress import make_legend_pickable

def main(paper=False):
    directory = "~/uplink/W_n_step_errors_test_dataset"
    directory = os.path.expanduser(directory)
    # list files in directory
    files = os.listdir(directory)
    errors = {}
    for file in files:
        if file.endswith("_n_step_errors.npz"):
            # load file
            data = np.load(os.path.join(directory, file))
            wm_name = file.split("_n_step_errors.npz")[0]
            obs_error = data["obs_error"]
            vecobs_error = data["vecobs_error"]
            errors[wm_name] = {"obs_error": obs_error, "vecobs_error": vecobs_error}

    if paper:
        fig, (ax1, ax2) = plt.subplots(1, 2, num="n_step_errors")
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        linegroups = []
        legends = []
        for wm_name in errors:
            obs_error = errors[wm_name]["obs_error"]
            vecobs_error = errors[wm_name]["vecobs_error"]
            # manual cleanup
            color = None
            style = None
            if wm_name == "RSSMWorldModel":
                continue
            if wm_name == "RSSMA0WorldModel":
                continue
            if wm_name == "GreyDummyWorldModel":
                color = "k"
                style = "--"
                wm_name = "Arbitrary-guess Upper Bound"
            if wm_name == "DummyWorldModel":
                color = "k"
                style = ":"
                wm_name = "Static Upper Bound"
            if wm_name == "GPT":
                wm_name = "Transformer (z=64)"
            if wm_name == "RSSMA0ExplicitWorldModel":
                wm_name = "RSSM (z=1024)"
            if wm_name == "TSSMWorldModel":
                wm_name = "TSSM (z=1024)"
            if wm_name == "TransformerLWorldModel":
                wm_name = "Transformer (z=1024)"
            line1, = ax1.plot(np.nanmean(obs_error, axis=0), color=color, linestyle=style)
            line2, = ax2.plot(np.nanmean(vecobs_error, axis=0), color=color, linestyle=style)
            linegroups.append([line1, line2])
            legends.append(wm_name)

        ax1.set_title("Image Observation")
        ax1.set_xlabel("Dream Length")
        ax1.set_ylabel("MSE Pixel-wise Prediction Error")
        ax2.set_title("Vector Observation")
        ax2.set_xlabel("Dream Length")
        ax2.set_ylabel("MSE Prediction Error [m]")
        ax1.set_xlim([0, 48])
        ax2.set_xlim([0, 48])
        L = fig.legend([lines[0] for lines in linegroups], legends, bbox_to_anchor=(0.9, 0.9))
        make_legend_pickable(L, linegroups)
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, num="errors")
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        linegroups = []
        legends = []
        for wm_name in errors:
            obs_error = errors[wm_name]["obs_error"]
            vecobs_error = errors[wm_name]["vecobs_error"]
            line1, = ax1.plot(np.nanmean(obs_error, axis=0))
            line2, = ax2.plot(np.nanmean(vecobs_error, axis=0))
            linegroups.append([line1, line2])
            legends.append(wm_name)
        L = fig.legend([lines[0] for lines in linegroups], legends)
        make_legend_pickable(L, linegroups)
        plt.show()


if __name__ == "__main__":
    StrictFire(main)
