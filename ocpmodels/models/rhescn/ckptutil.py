import torch


def filter_ckpt(state_dict: dict[str, torch.Tensor], prefix: str):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
