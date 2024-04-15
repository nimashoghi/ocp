import functools
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from .escn_compat import escn_compat_apply_tri_mask, eSCNCompatConfig
from .rhomboidal import Normalization, compute_idx_and_mask

log = getLogger(__name__)


@torch.no_grad()
def _grid_precomputes_optimized(
    mmax: int,
    res_beta: int,
    res_alpha: int,
    dtype: torch.dtype = torch.float,
    device: Optional[torch.device] = None,
    normalization: Normalization = "integral",
) -> tuple[
    Float[torch.Tensor, "beta alpha l_sq"],
    Float[torch.Tensor, "beta alpha l_sq"],
]:
    from e3nn.o3 import FromS2Grid, ToS2Grid

    if device is None:
        device = torch.device("cpu")

    lmax = 2 * mmax
    to_grid = ToS2Grid(
        lmax,
        (res_beta, res_alpha),
        normalization=normalization,
        device=device,
        dtype=dtype,
    )
    to_grid_sh_tri = torch.einsum("mbi,am->bai", to_grid.shb, to_grid.sha).detach()

    from_grid = FromS2Grid(
        (res_beta, res_alpha),
        lmax,
        normalization=normalization,
        device=device,
        dtype=dtype,
    )
    from_grid_sh_tri = torch.einsum(
        "am,mbi->bai", from_grid.sha, from_grid.shb
    ).detach()

    return to_grid_sh_tri, from_grid_sh_tri


def _grid_sh_tri_to_rh(
    sh: Float[torch.Tensor, "res_beta res_alpha l_sq"],
    mmax: int,
    *,
    stacked_m0: bool = False,
) -> Float[torch.Tensor, "res_beta res_alpha m 2 l"]:
    # l_sq = (lmax+1)**2
    lmax = 2 * mmax
    assert sh.shape[-1] == (lmax + 1) ** 2

    idx, mask = compute_idx_and_mask(mmax, stacked_m0=stacked_m0)
    sh_rh = sh[..., idx] * mask
    return sh_rh


class RhomboidalS2Grid(nn.Module):
    # Rhomboidal SO3 grid
    to_grid_sh_rh: Float[torch.Tensor, "res_beta*res_alpha m*2*l"]
    from_grid_sh_rh: Float[torch.Tensor, "res_beta res_alpha m 2 l"]

    # Triangular SO3 grid
    to_grid_sh_tri: Float[torch.Tensor, "res_beta*res_alpha l_sq"]
    from_grid_sh_tri: Float[torch.Tensor, "l_sq res_beta*res_alpha"]

    def __init__(
        self,
        mmax: int,
        escn_compat_config: Optional[eSCNCompatConfig] = None,
        res: Optional[Tuple[int, int]] = None,
        normalization: Normalization = "integral",
        grid_fp16: bool = False,
        stacked_m0: bool = False,
    ):
        super().__init__()

        def _convert(x: torch.Tensor) -> torch.Tensor:
            if grid_fp16:
                x = x.half()
            x = x.contiguous()
            return x

        self.mmax = mmax
        self.lmax = mmax * 2
        self.res_beta, self.res_alpha = res or (
            2 * (self.lmax + 1),
            2 * (self.mmax) + 1,
        )

        to_grid_sh_tri, from_grid_sh_tri = _grid_precomputes_optimized(
            mmax,
            self.res_beta,
            self.res_alpha,
            normalization=normalization,
        )
        (to_grid_sh_rh, from_grid_sh_rh) = map(
            functools.partial(_grid_sh_tri_to_rh, mmax=mmax, stacked_m0=stacked_m0),
            (to_grid_sh_tri, from_grid_sh_tri),
        )

        (
            to_grid_sh_rh,
            from_grid_sh_rh,
            to_grid_sh_tri,
            from_grid_sh_tri,
        ) = map(
            _convert,
            (to_grid_sh_rh, from_grid_sh_rh, to_grid_sh_tri, from_grid_sh_tri),
        )

        to_grid_sh_rh = rearrange(
            to_grid_sh_rh,
            "res_beta res_alpha m two l -> (res_beta res_alpha) (m two l)",
        ).contiguous()

        to_grid_sh_tri = rearrange(
            to_grid_sh_tri,
            "res_beta res_alpha l_sq -> (res_beta res_alpha) l_sq",
        ).contiguous()
        from_grid_sh_tri = rearrange(
            from_grid_sh_tri,
            "res_beta res_alpha l_sq -> l_sq (res_beta res_alpha)",
        ).contiguous()

        # ESCN compat masking
        to_grid_sh_tri = escn_compat_apply_tri_mask(
            escn_compat_config,
            to_grid_sh_tri,
            dim=1,
            with_mmax=False,
        )
        from_grid_sh_tri = escn_compat_apply_tri_mask(
            escn_compat_config,
            from_grid_sh_tri,
            dim=0,
            with_mmax=False,
        )

        self.register_buffer("to_grid_sh_tri", to_grid_sh_tri, persistent=False)
        self.register_buffer("from_grid_sh_tri", from_grid_sh_tri, persistent=False)

        self.register_buffer("to_grid_sh_rh", to_grid_sh_rh, persistent=False)
        if False:
            self.register_buffer("from_grid_sh_rh", from_grid_sh_rh, persistent=False)
