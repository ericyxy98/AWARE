import torch
import numpy as np
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class GroupSampler(Sampler[int]):
    r''' 
    By Xiangyu Yin
    
    Extending SubsetRandomSampler and BatchSampler with extra group information.
    Samples from same group will not be separated into different batches. This will 
    lead to varying batch sizes. The parameter batch_size_of_groups determines the
    number of groups in each batch.
    '''
    
    def __init__(self, indices: Sequence[int], groups: Sequence[int], batch_size_of_groups: int, generator=None) -> None:
        if not isinstance(batch_size_of_groups, int) or batch_size_of_groups <= 0:
            raise ValueError("batch_size_of_groups should be a positive integer value, "
                             "but got batch_size_of_groups={}".format(batch_size_of_groups))
        if len(indices) != len(groups):
            raise ValueError("Unequal length of indices and groups")
        
        self.indices = np.array(indices)
        self.groups = np.array(groups)
        self.batch_size_of_groups = batch_size_of_groups
        self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        unique_groups = list(set(self.groups))
        batch = []
        groups_in_batch = 0
        for i in torch.randperm(len(unique_groups), generator=self.generator):
            idx = self.indices[self.groups==unique_groups[i]]
            batch.extend(idx)
            groups_in_batch += 1
            if groups_in_batch == self.batch_size_of_groups:
                yield batch
                batch = []
                groups_in_batch = 0
        if groups_in_batch > 0:
            yield batch
    
    def __len__(self) -> int:
        unique_groups = list(set(self.groups))
        return len(unique_groups)//self.batch_size_of_groups+1