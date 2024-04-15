import functools
from typing import Dict, Literal, Tuple, cast

import numpy as np
import torch

Normalization = Literal["integral", "norm", "component"]


def signidx(x: int):
    return int(np.signbit(x).item())


@functools.cache
def _l_m_idxs(lmax: int):
    precomputed: Dict[Tuple[int, int], int] = {}
    precomputed_inv: Dict[int, Tuple[int, int]] = {}
    i = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            precomputed[(l, m)] = i
            precomputed_inv[i] = (l, m)
            i += 1

    def forward(l: int, m: int, none_if_not_found: bool = False):
        nonlocal precomputed
        if none_if_not_found:
            return cast(int, precomputed.get((l, m), None))
        return precomputed[(l, m)]

    def inverse(i: int, none_if_not_found: bool = False):
        nonlocal precomputed_inv
        if none_if_not_found:
            return cast(Tuple[int, int], precomputed_inv.get(i, None))
        return precomputed_inv[i]

    return forward, inverse


def compute_idx_and_mask(
    mmax: int,
    *,
    stacked_m0: bool = False,
):
    lmax = 2 * mmax

    idx = torch.zeros((mmax + 1, 2, mmax + 1), dtype=torch.long)

    idx_fn, _ = _l_m_idxs(lmax)
    mask = torch.ones_like(idx, dtype=torch.bool)
    for m in range(mmax + 1):
        for l_idx, l in enumerate(range(m, m + mmax + 1)):
            # If l > lmax, then we need to mask out the index and continue
            if l > lmax:
                mask[m, :, l_idx] = False
                continue

            idx[m, signidx(+1), l_idx] = idx_fn(l, m)
            # If m > 0, we set the `-m` indices.
            if m > 0:
                idx[m, signidx(-1), l_idx] = idx_fn(l, -m)
                continue

            assert m == 0
            # Otherwise (m == 0), we set the negative indices to be
            #  the (mmax + |l|)th index if `stacked_m0` is True, otherwise
            #  we mask them out.
            if stacked_m0:
                idx[m, signidx(-1), l_idx] = idx_fn(l + mmax, m)
            else:
                mask[m, signidx(-1), l_idx] = False

    return idx, mask
