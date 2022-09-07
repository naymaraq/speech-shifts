import torch
import numpy as np

import speech_shifts.common.torch_scatter as torch_scatter

def numel(obj):
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")

def avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    
    assert v.device==g.device
    assert v.numel()==g.numel()
    group_count = get_counts(g, n_groups)
    group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    return group_avgs, group_count

def get_counts(g, n_groups):
    """
    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - counts (Tensor): A list of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts

def minimum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmin(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return min(numbers)

def maximum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmax(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return max(numbers)

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts