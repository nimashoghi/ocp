from typing import NamedTuple

import torch


class eSCNCompatConfig(NamedTuple):
    lmax: int | None
    mmax: int | None


def escn_compat_apply_tri_mmax_mask(
    config: eSCNCompatConfig | None,
    x: torch.Tensor,
    dim: int = 1,
):
    if config is None or config.mmax is None:
        return x

    # X is shape (E, (lmax+1)^2, C)
    # Let's get the lmax
    lmax = int((x.shape[dim]) ** 0.5) - 1

    i = 0
    mask = torch.ones((x.shape[dim]), dtype=torch.bool, device=x.device)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if abs(m) > config.mmax:
                mask[i] = False
            i += 1

    #  Add Nones to the mask to make it the right shape
    for _ in range(dim):
        mask = mask.unsqueeze(0)
    for _ in range(dim + 1, x.dim()):
        mask = mask.unsqueeze(-1)

    x = x * mask
    return x


def escn_compat_apply_tri_mask(
    config: eSCNCompatConfig | None,
    x: torch.Tensor,
    dim: int = 1,
    with_mmax: bool = False,
):
    if config is None or config.lmax is None:
        return x
    count = (config.lmax + 1) ** 2

    idx = ()
    for _ in range(dim):
        idx += (slice(None),)

    idx_to_keep = idx + (slice(count),)
    idx_to_zero = idx + (slice(count, None),)

    x = torch.cat([x[idx_to_keep], torch.zeros_like(x[idx_to_zero])], dim=dim)

    if with_mmax:
        x = escn_compat_apply_tri_mmax_mask(config, x, dim=dim)
    return x
