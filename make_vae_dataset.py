import os
from navrep3denv import NavRep3DEnv
from navrep.tools.commonargs import parse_multiproc_args
from navrep.scripts.make_vae_dataset import generate_vae_dataset, RandomMomentumPolicy

if __name__ == "__main__":
    args, _ = parse_multiproc_args()
    n_sequences = 100
    if args.n is not None:
        n_sequences = args.n

    archive_dir = os.path.expanduser("~/navrep3d/datasets/V/navrep3dtrain")
    if args.dry_run:
        archive_dir = "/tmp/navrep3d/datasets/V/navrep3dtrain"
    env = NavRep3DEnv(verbose=0, collect_statistics=False)
    generate_vae_dataset(
        env, n_sequences=n_sequences,
        subset_index=args.subproc_id, n_subsets=args.n_subprocs,
        policy=RandomMomentumPolicy(),
        render=args.render, archive_dir=archive_dir)
