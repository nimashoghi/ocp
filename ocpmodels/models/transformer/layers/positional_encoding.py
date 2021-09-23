import torch


def calculate_relative_positions(positions: torch.Tensor):
    """
    Calculate relative positions of the given positions.
    Takes an input of shape [batch_size, n_dimensions]
    Returns a tensor of shape [batch_size, batch_size, n_dimensions]

    It gets the euclidean vector between each pair of positions and stores it in the output tensor -- diagonal elements of output are zero.

    For example,
    >>> calculate_relative_positions(torch.tensor([[1, 2, 3], [4, 4, 4]]))
    tensor([[[ 0,  0,  0],
             [-3, -2, -1]],
    <BLANKLINE>
            [[ 3,  2,  1],
             [ 0,  0,  0]]])
    """
    return positions[:, None, :] - positions[None, :, :]


#%%
positions = torch.tensor(
    [
        [1, 2, 3],
        [4, 4, 4],
    ]
)

# calculate_relative_positions(positions)
# #%%
# positions.shape
# positions[:, None, :]
# positions[None, :, :]

#%%
positions.shape
positions[:, None, :].shape
positions[None, :, :]
