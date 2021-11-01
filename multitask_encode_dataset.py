import numpy as np
import os
from tqdm import tqdm
from strictfire import StrictFire

from navrep3d.encodedenv3d import EnvEncoder

_RS = 5
_H = 64
encoder_types = ["E2E", "N3D", "sequenceN3D"]

def main(dry_run=False):
    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")

    filenames = []
    for dirpath, dirnames, dirfilename in os.walk(archive_dir):
        for filename in [
            f
            for f in dirfilename
            if f.endswith("images_labels.npz")
        ]:
            filenames.append(os.path.join(dirpath, filename))
    for archive_file in filenames:
        archive_path = os.path.join(archive_dir, archive_file)
        data = np.load(archive_path)
        print("{} loaded.".format(archive_path))
        images = data["images"]
        labels = data["labels"]
        depths = data["depths"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        robotstates = data["robotstates"]

        for encoder_type in encoder_types:
            if encoder_type == "E2E":
                # img, rs -> 64
                modelpath = os.path.expanduser("~/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip") # noqa
                encoder = EnvEncoder("E2E", "V_ONLY", gpt_model_path=modelpath)
            elif encoder_type == "N3D":
                # img, rs -> 64
                # (can use z)
                modelpath = os.path.expanduser("~/navrep3d_W/models/W/transformer_Salt")
                encoder = EnvEncoder("GPT", "V_ONLY", gpt_model_path=modelpath)
            elif encoder_type == "sequenceN3D":
                # {(img, a, rs, d), (img, a, rs, d), ...} -> (64, 64, 64, 64)
                # (uses get_h)
                modelpath = os.path.expanduser("~/navrep3d_W/models/W/transformer_Salt")
                encoder = EnvEncoder("GPT", "M_ONLY", gpt_model_path=modelpath)
            else:
                raise NotImplementedError

            encodings = []
            for image, label, depth, action, done, robotstate in tqdm(zip(
                    images, labels, depths, actions, dones, robotstates)):
                obs = (image, robotstate)
                h = encoder._encode_obs(obs, action)
                h_no_rs = h[:-_RS]
                assert h_no_rs.shape == (_H,)
                if done:
                    encoder.reset()
                encodings.append(h_no_rs)
            encodings = np.array(encodings)
            data = dict(encodings=encodings, labels=labels, depths=depths,
                        robotstates=robotstates, actions=actions, rewards=rewards, dones=dones,
                        modelpath=np.array(modelpath))
            write_path = archive_path.replace("_images_", "_{}encodings_".format(encoder_type))
            if dry_run:
                write_path = os.path.join("/tmp", os.path.basename(write_path))
            np.savez_compressed(write_path, **data)
            print(write_path, "written.")


if __name__ == "__main__":
    StrictFire(main)
