import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import typer
from navrep.models.gpt import load_checkpoint

from multitask_train import onehot_to_rgb, TaskLearner, MultitaskDataset

_RS = 5
_H = 64
N_CLASSES = 6

BATCH_SIZE = 128
_5 = 5

class ComparisonDataset(object):
    def __init__(self, directory, **kwargs):
        self.baseline_depth_dataset = MultitaskDataset(directory, "depth", from_image=True,
                                                       filename_mask="_images_labels.npz", **kwargs)
        self.E2E_labels_dataset = MultitaskDataset(directory, "segmentation", from_image=False,
                                                   filename_mask="_E2Eencodings_labels.npz", **kwargs)
        self.N3D_labels_dataset = MultitaskDataset(directory, "segmentation", from_image=False,
                                                   filename_mask="_N3Dencodings_labels.npz", **kwargs)

    def __len__(self):
        return len(self.baseline_depth_dataset.data["labels"])

    def __getitem__(self, idx):
        img, depth = self.baseline_depth_dataset[idx]
        E2Eencoding, label = self.E2E_labels_dataset[idx]
        N3Dencoding, _ = self.N3D_labels_dataset[idx]
        return img, E2Eencoding, N3Dencoding, label, depth

def find_checkpoints(archive_dir, encoder_type, task):
    filenames = []
    if task == "segmentation":
        task = "segmenter"
    filename_mask = "{}_{}_".format(encoder_type, task)
    for dirpath, dirnames, dirfilename in os.walk(archive_dir):
        for filename in [
            f
            for f in dirfilename
            if f.startswith(filename_mask)
        ]:
            filenames.append(os.path.join(dirpath, filename))
    return sorted(filenames)

def main(gpu : bool = False):
    tasks = ["segmentation", "depth"]
    encoder_types = ["baseline", "E2E", "N3D"]
    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_comparison")
    dataset = ComparisonDataset(archive_dir)
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    # get first batch
    for img, E2Eencoding, N3Dencoding, label, depth in loader:
        break

    models_dir = os.path.expanduser("~/navrep3d/models/multitask")
    predictions = {}
    for encoder_type in encoder_types:
        predictions[encoder_type] = {}
        for task in tasks:
            label_is_onehot = task == "segmentation"
            task_channels = N_CLASSES if label_is_onehot else 1
            from_image = encoder_type == "baseline"
            model = TaskLearner(task_channels, from_image, label_is_onehot, gpu=gpu)
            model_checkpoints = find_checkpoints(models_dir, encoder_type, task)
            if len(model_checkpoints) > 1:
                print("More than one checkpoint found for {} {} encoder.".format(encoder_type, task))
            if len(model_checkpoints) == 0:
                raise ValueError("No checkpoint found for {} {} encoder.".format(encoder_type, task))
            checkpoint = model_checkpoints[0]
            load_checkpoint(model, checkpoint, gpu=gpu)
            # Predictions
            model.train(False)
            if encoder_type == "baseline":
                x = img
            elif encoder_type == "N3D":
                x = N3Dencoding
            elif encoder_type == "E2E":
                x = E2Eencoding
            else:
                raise NotImplementedError
            with torch.set_grad_enabled(False):
                y_pred, _ = model(x)
            predictions[encoder_type][task] = y_pred

    # save plot
    from matplotlib import pyplot as plt
    plt.figure("predictions")
    plt.clf()
    plt.suptitle("Comparison of predictions")
    ROW = 9
    f, axes = plt.subplots(ROW, _5, num="predictions", sharex=True, sharey=True)
    axes = axes.reshape((ROW, _5))
    for i, axrow in enumerate(axes.T):
        ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axrow
        ax0.imshow(np.moveaxis(img.cpu().numpy()[i], 0, -1))
        ax1.imshow(onehot_to_rgb(np.moveaxis(label.cpu().numpy()[i], 0, -1)))
        ax2.imshow(onehot_to_rgb(np.moveaxis(
            predictions["baseline"]["segmentation"].detach().cpu().numpy()[i], 0, -1)))
        ax3.imshow(onehot_to_rgb(np.moveaxis(
            predictions["E2E"]["segmentation"].detach().cpu().numpy()[i], 0, -1)))
        ax4.imshow(onehot_to_rgb(np.moveaxis(
            predictions["N3D"]["segmentation"].detach().cpu().numpy()[i], 0, -1)))
        ax5.imshow(np.moveaxis(depth.cpu().numpy()[i], 0, -1))
        ax6.imshow(np.moveaxis(predictions["baseline"]["depth"].detach().cpu().numpy()[i], 0, -1))
        ax7.imshow(np.moveaxis(predictions["E2E"]["depth"].detach().cpu().numpy()[i], 0, -1))
        ax8.imshow(np.moveaxis(predictions["N3D"]["depth"].detach().cpu().numpy()[i], 0, -1))
        if i == 0:
            ax0.set_ylabel("input")
            ax1.set_ylabel("GT")
            ax2.set_ylabel("specific")
            ax3.set_ylabel("E2E")
            ax4.set_ylabel("N3D")
            ax5.set_ylabel("GT")
            ax6.set_ylabel("specific")
            ax7.set_ylabel("E2E")
            ax8.set_ylabel("N3D")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
