import copy

import numpy as np
import torch
from speech_shifts.common.utils import get_counts
import warnings

class Grouper:
    """
    Groupers group data points together based on their metadata.
    They are used for training and evaluation,
    e.g., to measure the accuracies of different groups of data.
    """
    def __init__(self):
        raise NotImplementedError

    @property
    def n_groups(self):
        """
        The number of groups defined by this Grouper.
        """
        return self._n_groups

    def metadata_to_group(self, metadata, return_counts=False):
        """
        Args:
            - metadata (Tensor): An n x d matrix containing d metadata fields
                                 for n different points.
            - return_counts (bool): If True, return group counts as well.
        Output:
            - group (Tensor): An n-length vector of groups.
            - group_counts (Tensor): Optional, depending on return_counts.
                                     An n_group-length vector of integers containing the
                                     numbers of data points in each group in the metadata.
        """
        raise NotImplementedError

    def group_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the pretty name of that group.
        """
        raise NotImplementedError

    def group_field_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the name of that group.
        """
        raise NotImplementedError

class CombinatorialGrouper(Grouper):
    def __init__(self, meta_fields, meta_map, meta_array, groupby_fields):

        metadata_fields = copy.deepcopy(meta_fields)
        largest_metadata_map = copy.deepcopy(meta_map)
        self.groupby_fields = groupby_fields

        if groupby_fields is None:
            self._n_groups = 1
        else:
            self.groupby_field_indices = [i for (i, field) in enumerate(metadata_fields) if field in groupby_fields]
            if len(self.groupby_field_indices) != len(self.groupby_fields):
                raise ValueError('At least one group field not found in dataset.metadata_fields')

            metadata_array = torch.clone(meta_array)
            grouped_metadata = metadata_array[:, self.groupby_field_indices]
            
            if not isinstance(grouped_metadata, torch.LongTensor):
                grouped_metadata_long = grouped_metadata.long()
                if not torch.all(grouped_metadata == grouped_metadata_long):
                    warnings.warn(f'CombinatorialGrouper: converting metadata with fields [{", ".join(groupby_fields)}] into long')
                grouped_metadata = grouped_metadata_long

            for idx, field in enumerate(self.groupby_fields):
                min_value = grouped_metadata[:,idx].min()
                if min_value < 0:
                    raise ValueError(f"Metadata for CombinatorialGrouper cannot have values less than 0: {field}, {min_value}")
                if min_value > 0:
                    warnings.warn(f"Minimum metadata value for CombinatorialGrouper is not 0 ({field}, {min_value}). This will result in empty groups")

            # We assume that the metadata fields are integers,
            # so we can measure the cardinality of each field by taking its max + 1.
            # Note that this might result in some empty groups.
            assert grouped_metadata.min() >= 0, "Group numbers cannot be negative."
            self.cardinality = 1 + torch.max(grouped_metadata, dim=0)[0]
            cumprod = torch.cumprod(self.cardinality, dim=0)
            self._n_groups = cumprod[-1].item()
            self.factors_np = np.concatenate(([1], cumprod[:-1]))
            self.factors = torch.from_numpy(self.factors_np)
            self.metadata_map = largest_metadata_map


    def metadata_to_group(self, metadata, return_counts=False):
        if self.groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = metadata[:, self.groupby_field_indices].long() @ self.factors

        if return_counts:
            group_counts = get_counts(groups, self._n_groups)
            return groups, group_counts
        else:
            return groups

    def group_str(self, group):
        if self.groupby_fields is None:
            return 'all'

        # group is just an integer, not a Tensor
        n = len(self.factors_np)
        metadata = np.zeros(n)
        for i in range(n-1):
            metadata[i] = (group % self.factors_np[i+1]) // self.factors_np[i]
        metadata[n-1] = group // self.factors_np[n-1]
        group_name = ''
        for i in reversed(range(n)):
            meta_val = int(metadata[i])
            if self.metadata_map is not None:
                if self.groupby_fields[i] in self.metadata_map:
                    meta_val = self.metadata_map[self.groupby_fields[i]][meta_val]
            group_name += f'{self.groupby_fields[i]} = {meta_val}, '
        group_name = group_name[:-2]
        return group_name

    def group_field_str(self, group):
        return self.group_str(group).replace('=', ':').replace(',','_').replace(' ','')