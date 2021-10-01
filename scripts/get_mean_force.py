#!/usr/bin/env python

import argparse
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.datasets import TrajectoryLmdbDataset

dataset_paths = {
    "train": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/all/",
    "val": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/id_30k/",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic_numbers", nargs="+", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="val")
    parser.add_argument("--out", type=str)

    args = parser.parse_args()
    assert args.dataset in [
        "train",
        "val",
    ], "Invalid dataset - must be train or val"

    dataset = TrajectoryLmdbDataset(dict(src=dataset_paths[args.dataset]))
    collate_fn = ParallelCollater(0, False)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate_fn,
        num_workers=64,
    )

    means_dict = defaultdict(list)
    for [data] in tqdm(dataloader):
        for atomic_number in args.atomic_numbers:
            forces = data.force[data.atomic_numbers == atomic_number]
            means_dict[atomic_number].append(forces.mean(dim=0))

    mean_of_means = {
        atomic_number: torch.stack(means).mean(dim=0)
        for atomic_number, means in means_dict.items()
    }
    print(mean_of_means)
    if args.out:
        torch.save(mean_of_means, args.out)


if __name__ == "__main__":
    main()
