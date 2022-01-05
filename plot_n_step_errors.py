from matplotlib import pyplot as plt
import numpy as np
import os

from plot_gym_training_progress import make_legend_pickable

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

fig, (ax1, ax2) = plt.subplots(2, 1, num="errors")
# ax1.set_yscale('log')
# ax2.set_yscale('log')
linegroups = []
legends = []
for wm_name in errors:
    # manual cleanup
    if wm_name == "RSSMWorldModel":
        continue
    obs_error = errors[wm_name]["obs_error"]
    vecobs_error = errors[wm_name]["vecobs_error"]
    line1, = ax1.plot(np.nanmean(obs_error, axis=0))
    line2, = ax2.plot(np.nanmean(vecobs_error, axis=0))
    linegroups.append([line1, line2])
    legends.append(wm_name)
L = fig.legend([lines[0] for lines in linegroups], legends)
make_legend_pickable(L, linegroups)
plt.show()
