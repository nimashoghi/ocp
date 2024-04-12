import functools
from typing import Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import torch
from e3nn.o3 import spherical_harmonics_s2_grid
from e3nn.o3._s2grid import _quadrature_weights
from einops import rearrange
from jaxtyping import Float


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


def compute_idx_and_mask(mmax: int):
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
            # If m == 0, we set the negative indices to be
            #  the (mmax + |l|)th index
            if m == 0:
                idx[m, signidx(-1), l_idx] = idx_fn(l + mmax, m)
            # Otherwise (m > 0), we set the `-m` indices.
            else:
                idx[m, signidx(-1), l_idx] = idx_fn(l, -m)

    # Mask out m=0 negative indices
    for l in range(mmax):
        # mask = mask.at[0, signidx(-1), :].set(False)
        mask[0, signidx(-1), :] = False

    return idx, mask


# Vectorize over `E` dimension
# @functools.partial(torch.vmap, in_dims=(0, None, None), out_dims=0)
def full_wigner_to_sparse(
    full_wigner: Float[torch.Tensor, "(lmax+1)**2 (lmax+1)**2"],
    rh_idx: torch.Tensor,
    rh_mask: torch.Tensor,
):
    # full_wigner.shape: (lmax+1)**2, (lmax+1)**2

    wigner = full_wigner[..., rh_idx] * rh_mask[None, None]
    # wigner.shape: (lmax+1)**2, (mmax+1), 2, (mmax+1)
    wigner_inv = full_wigner.transpose(-1, -2)[:, rh_idx] * rh_mask[None, ..., None]
    # wigner_inv.shape: (mmax+1), 2, (mmax+1), (lmax+1)**2

    return wigner, wigner_inv


def _rollout_sh(input: torch.Tensor, lmax: int) -> torch.Tensor:
    """
    Input:
        [[(0,0)            ]       l=0
         [(1,0) (1,1)      ]       l=1
         [(2,0) (2,1) (2,2)]]      l=2
    Output:
        [(0,0) (1,1) (1,0) (1,1) (2,2) (2,1) (2,0) (2,1) (2,2)]
    """
    assert input.shape[-2] == lmax + 1  # l
    assert input.shape[-1] == lmax + 1  # abs(m)
    ls = []
    ms = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ls.append(l)
            ms.append(abs(m))
    ls = torch.tensor(ls)
    ms = torch.tensor(ms)
    return input[..., ls, ms]


def _normalization(
    lmax: int,
    normalization: str,
    direction: str,
    dtype: np.dtype = np.float32,
    lmax_in: Optional[int] = None,
) -> np.ndarray:
    """Handles normalization of different components of IrrepsArrays."""
    assert direction in ["to_s2", "from_s2"]

    if normalization == "component":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1
        if direction == "to_s2":
            return np.sqrt(4 * np.pi) / (
                (np.sqrt(2 * np.arange(lmax + 1) + 1)).astype(dtype) * np.sqrt(lmax + 1)
            )
        else:
            return np.sqrt(4 * np.pi) * (
                (np.sqrt(2 * np.arange(lmax + 1) + 1)).astype(dtype) * np.sqrt(lmax + 1)
            )
    if normalization == "norm":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1/(2L+1)
        if direction == "to_s2":
            return np.sqrt(4 * np.pi) * np.ones(lmax + 1, dtype) / np.sqrt(lmax + 1)
        else:
            return np.sqrt(4 * np.pi) * np.ones(lmax + 1, dtype) * np.sqrt(lmax_in + 1)
    if normalization == "integral":
        # normalize such that the coefficient L=0 is equal to 4 pi the integral of the function
        # for "integral" normalization, the direction does not matter.
        return np.ones(lmax + 1, dtype) * np.sqrt(4 * np.pi)

    raise Exception("normalization needs to be 'norm', 'component' or 'integral'")


def _expand_matrix(ls: List[int]) -> np.ndarray:
    """
    conversion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)
    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    Args:
        ls: list of l values
    Returns:
        array of shape ``[l, m, l * m]``
    """
    lmax = max(ls)
    m = np.zeros((lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in ls)), np.float64)
    i = 0
    for l in ls:
        m[l, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(
            2 * l + 1, dtype=np.float64
        )
        i += 2 * l + 1
    return m


Normalization = Literal["integral", "norm", "component"]


def to_grid_precomputes(
    mmax: int,
    res_beta: int,
    res_alpha: int,
    dtype: torch.dtype = torch.float,
    device: Optional[torch.device] = None,
    quadrature: Literal["soft"] = "soft",
    normalization: Normalization = "integral",
):
    assert quadrature == "soft", "Only soft quadrature is supported for now"
    if quadrature == "soft":
        assert res_beta % 2 == 0, "res_beta must be even for soft quadrature"

    if device is None:
        device = torch.device("cpu")

    lmax = 2 * mmax
    _, _, sh_beta, sh_alpha = spherical_harmonics_s2_grid(
        lmax,
        res_beta,
        res_alpha,
        device=device,
        dtype=dtype,
    )
    sh_beta = rearrange(
        sh_beta, "res_beta (l m) -> res_beta l m", l=lmax + 1, m=lmax + 1
    )
    # sh_beta shape: res_beta lmax[l] lmax[m]
    # sh_alpha shape: res_alpha (2 * lmax + 1)

    n = _normalization(lmax, normalization, "to_s2")
    n = torch.from_numpy(n).to(device=device, dtype=dtype)
    # ^ Shape: lmax

    sh_beta = torch.einsum("tlm,l->tlm", sh_beta, n)
    # ^ Shape: res_beta lmax[l] lmax[m]

    # Roll out to flattened triangular form
    sh_beta = _rollout_sh(sh_beta, lmax)
    # ^ Shape: res_beta (lmax+1)**2

    # Triangular form
    m = torch.from_numpy(_expand_matrix(list(range(lmax + 1)))).to(
        device, dtype
    )  # (lmax+1) 2lmax+1 (lmax+1)**2
    sh_beta_triangular = torch.einsum(
        "lmj,bj,lmi->mbi", m, sh_beta, m
    )  # (lmax+1) res_beta (lmax+1)**2
    sh_alpha_triangular = sh_alpha

    # Now index into sh_beta to get the rhomboidal form
    idx, mask = compute_idx_and_mask(mmax)
    sh_beta_rhomboidal = sh_beta[..., idx] * mask

    # Convert sh_alpha to rhomboidal form
    sh_alpha_idx = torch.zeros((mmax + 1, 2), dtype=torch.long, device=device)
    for i, m in enumerate(range(-lmax, lmax + 1)):
        if abs(m) > mmax:
            continue
        # sh_alpha_idx = sh_alpha_idx.at[abs(m), signidx(m)].set(i)
        sh_alpha_idx[abs(m), signidx(m)] = i

    sh_alpha_rhomboidal = sh_alpha[..., sh_alpha_idx]
    # ^ Shape: res_alpha mmax[m] 2

    return (
        (sh_beta_triangular, sh_alpha_triangular),
        (sh_beta_rhomboidal, sh_alpha_rhomboidal),
    )


def from_grid_precomputes(
    mmax: int,
    res_beta: int,
    res_alpha: int,
    dtype: torch.dtype = torch.float,
    device: Optional[torch.device] = None,
    quadrature: Literal["soft"] = "soft",
    normalization: Normalization = "integral",
):
    assert quadrature == "soft", "Only soft quadrature is supported for now"
    if quadrature == "soft":
        assert res_beta % 2 == 0, "res_beta must be even for soft quadrature"

    if device is None:
        device = torch.device("cpu")

    lmax = 2 * mmax
    _, _, sh_beta, sh_alpha = spherical_harmonics_s2_grid(
        lmax,
        res_beta,
        res_alpha,
        device=device,
        dtype=dtype,
    )
    sh_beta = rearrange(
        sh_beta, "res_beta (l m) -> res_beta l m", l=lmax + 1, m=lmax + 1
    )
    qw = (
        _quadrature_weights(res_beta // 2, dtype=dtype, device=device)
        * res_beta**2
        / res_alpha
    )  # [b]

    # sh_beta shape: res_beta lmax[l] lmax[m]
    # sh_alpha shape: res_alpha (2 * lmax + 1)
    # qw shape: res_beta

    sh_alpha = sh_alpha / res_alpha

    n = _normalization(lmax, normalization, "from_s2")
    n = torch.from_numpy(n).to(device=device, dtype=dtype)
    # ^ Shape: lmax

    sh_beta = torch.einsum("tml,l,t->tml", sh_beta, n, qw)
    # ^ Shape: res_beta lmax[l] lmax[m]

    # Roll out to flattened triangular form
    sh_beta = _rollout_sh(sh_beta, lmax)

    # Triangular form
    m = torch.from_numpy(_expand_matrix(list(range(lmax + 1)))).to(
        device, dtype
    )  # (lmax+1) 2lmax+1 (lmax+1)**2
    sh_beta_triangular = torch.einsum(
        "lmj,bj,lmi->mbi", m, sh_beta, m
    )  # (lmax+1) res_beta (lmax+1)**2
    sh_alpha_triangular = sh_alpha

    # Now index into sh_beta to get the rhomboidal form
    idx, mask = compute_idx_and_mask(mmax)
    sh_beta_rhomboidal = sh_beta[..., idx] * mask

    # Convert sh_alpha to rhomboidal form
    sh_alpha_idx = torch.zeros((mmax + 1, 2), dtype=torch.long, device=device)
    for i, m in enumerate(range(-lmax, lmax + 1)):
        if abs(m) > mmax:
            continue
        # sh_alpha_idx = sh_alpha_idx.at[abs(m), signidx(m)].set(i)
        sh_alpha_idx[abs(m), signidx(m)] = i

    sh_alpha_rhomboidal = sh_alpha[..., sh_alpha_idx]
    # ^ Shape: res_alpha mmax[m] 2

    return (
        (sh_beta_triangular, sh_alpha_triangular),
        (sh_beta_rhomboidal, sh_alpha_rhomboidal),
    )
