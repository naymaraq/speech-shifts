import torch

def _se_pool_step_script_train(x: torch.Tensor, context_window: int, mask: torch.Tensor):
    """
    Calculates the masked average over padded limited context segment during training mode.
    Randomly slices a segment of length `context_window` from signal+padded input tensor across all channels and
    uses it for computing masked limited context.
    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].
    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(x.dtype)
    else:
        start_idx = torch.randint(0, timesteps - context_window, size=[1], dtype=torch.int32)[0]
        x = x[:, :, start_idx : (start_idx + context_window)]  # [B, C, context_window]
        mask = mask[:, :, start_idx : (start_idx + context_window)]  # [B, 1, context_window]

        mask = mask.sum(dim=-1, keepdim=True).to(x.dtype)  # [B, C, 1]
        y = x.sum(dim=-1, keepdim=True)  # [B, 1, 1]
        y = y / (mask + 1e-8)  # [B, C, 1]

    return y


def _se_pool_step_script_infer(x: torch.Tensor, context_window: int, mask: torch.Tensor):
    """
    Calculates the masked average over padded limited context segment during inference mode.
    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].
    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(x.dtype)
    else:
        # << During inference prefer to use entire context >>
        # x = x[:, :, :context_window]  # [B, C, context_window]
        # mask = mask[:, :, :context_window]  # [B, 1, context_window]
        #
        # mask = mask.sum(dim=-1, keepdim=True).to(x.dtype)  # [B, C, 1]
        # y = x.sum(dim=-1, keepdim=True)  # [B, 1, 1]
        # y = y / (mask + 1e-8)  # [B, C, 1]
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(x.dtype)

    return y