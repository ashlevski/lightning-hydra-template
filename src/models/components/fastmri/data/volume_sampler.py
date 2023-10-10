"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from fastmri.data.mri_data import CombinedSliceDataset, SliceDataset


class VolumeSampler(Sampler):
    """
    Sampler for volumetric MRI data.

    Based on pytorch DistributedSampler, the difference is that all instances
    from the same MRI volume need to go to the same node for distributed
    training. Dataset example is a list of tuples (fname, instance), where
    fname is essentially the volume name (actually a filename).
    """

    def __init__(
        self,
        dataset: Union[CombinedSliceDataset, SliceDataset],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: An MRI dataset (e.g., SliceData).
            num_replicas: Number of processes participating in distributed
                training. By default, :attr:`rank` is retrieved from the
                current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`. By
                default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle: If ``True`` (default), sampler will shuffle the indices.
            seed: random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across
                all processes in the distributed group.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # get all file names and split them based on number of processes
        self.all_volume_names = sorted(
            set(str(raw_sample[0]) for raw_sample in self.dataset.raw_samples)
        )
        self.all_volumes_split: List[List[str]] = []
        for rank_num in range(self.num_replicas):
            self.all_volumes_split.append(
                [
                    self.all_volume_names[i]
                    for i in range(
                        rank_num, len(self.all_volume_names), self.num_replicas
                    )
                ]
            )

        # get slice indices for each file name
        rank_indices: List[List[int]] = [[] for _ in range(self.num_replicas)]
        for i, raw_sample in enumerate(self.dataset.raw_samples):
            vname = str(raw_sample[0])
            for rank_num in range(self.num_replicas):
                if vname in self.all_volumes_split[rank_num]:
                    rank_indices[rank_num].append(i)
                    break

        # need to send equal number of samples to each process - take the max
        self.num_samples = max(len(indices) for indices in rank_indices)
        self.total_size = self.num_samples * self.num_replicas
        self.indices = rank_indices[self.rank]

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in ordering]
        else:
            indices = self.indices

        # add extra samples to match num_samples
        repeat_times = self.num_samples // len(indices)
        indices = indices * repeat_times
        indices = indices + indices[: self.num_samples - len(indices)]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
