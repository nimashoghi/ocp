import torch
from typing import Optional
from torch.nn.utils.rnn import pad_sequence


def convert_graph_to_tensor(batch, device: Optional[torch.device] = None):
    atomic_numbers = pad_sequence([x.atomic_numbers.long() for x in batch])
    positions = pad_sequence([x.pos for x in batch])
    key_padding_mask = atomic_numbers != 0
    if device is not None:
        atomic_numbers = atomic_numbers.to(device)
        positions = positions.to(device)
        key_padding_mask = key_padding_mask.to(device)
    return atomic_numbers, positions, key_padding_mask
