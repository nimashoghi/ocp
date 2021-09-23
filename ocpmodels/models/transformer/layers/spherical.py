import torch


def convert_to_spherical(input: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of 3d cartesian coordinates to spherical coordinates.
    :param input: A tensor of shape (batch_size, num_points, 3)
    :return: A tensor of shape (batch_size, num_points, 3)
    """
    r = input.norm(dim=-1)
    theta = torch.acos(input[..., 2] / r)
    phi = torch.atan2(input[..., 1], input[..., 0])
    return torch.stack([r, theta, phi], dim=-1)
