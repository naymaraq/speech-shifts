import torch

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