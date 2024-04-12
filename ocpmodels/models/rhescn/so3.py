"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
from collections.abc import Callable
from logging import getLogger
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from ll import ActSave

from .rhomboidal import Normalization, compute_idx_and_mask

if TYPE_CHECKING:
    from .model import Masker

log = getLogger(__name__)


@torch.jit.script
def fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_optimized(
    x: torch.Tensor,  # E (m 2 l) C
    to_grid_sh_rh: torch.Tensor,  # (res_beta res_alpha) (m 2 l)
    wigner_inv_from_grid: torch.Tensor,  # E (res_beta res_alpha) l_sq
):
    # x = torch.tensordot(x, to_grid_sh_rh, dims=((1, 3, 2), (2, 4, 3)))
    # x = F.silu(x)
    # x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), wigner_inv_from_grid)
    # x = x.transpose(-1, -2)
    # return x
    x = torch.bmm(to_grid_sh_rh[None].expand(x.shape[0], -1, -1), x)
    x = F.silu(x)
    x = torch.bmm(wigner_inv_from_grid, x)
    return x


def fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_unoptimized(
    x: torch.Tensor,  # E (m 2 l) C
    to_grid_sh_rh: torch.Tensor,  # (res_beta res_alpha) (m 2 l)
    from_grid_sh_tri: Float[torch.Tensor, "l_sq res_beta*res_alpha"],
    wigner_inv_full: Float[torch.Tensor, "E l_sq l_sq"],
    masker: "Masker",
):
    with ActSave.context("act_rotate_inv"):
        # x = torch.tensordot(x, to_grid_sh_rh, dims=((1, 3, 2), (2, 4, 3)))
        # x = F.silu(x)
        # x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), wigner_inv_from_grid)
        # x = x.transpose(-1, -2)
        # return x
        x = torch.bmm(to_grid_sh_rh[None].expand(x.shape[0], -1, -1), x)
        ActSave({"x_grid": x})
        # ^ Shape: E res_beta res_alpha C
        x = F.silu(x)
        ActSave({"x_grid_silu": x})
        # ^ Shape: E res_beta res_alpha C
        # from_grid_sh_tri = masker(from_grid_sh_tri, dim=0, with_mmax=True)
        x = torch.bmm(from_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
        ActSave({"x_sphere_silu": x})
        # ^ Shape: E l_sq C
        # x = masker(x, dim=1)
        wigner_inv_full = masker(wigner_inv_full, dim=1, with_mmax=False)
        wigner_inv_full = masker(wigner_inv_full, dim=2, with_mmax=True)
        x = torch.bmm(wigner_inv_full, x)
        ActSave({"x_sphere_rotated": x})
        # ^ Shape: E l_sq C
        return x


@torch.jit.script
def fused_s2_pointwise_conv_optimized(
    x: torch.Tensor,  # N l_sq C,
    x_message: torch.Tensor,  # N l_sq C,
    to_grid_sh_tri: torch.Tensor,  # res_beta*res_alpha l_sq
    from_grid_sh_tri: torch.Tensor,  # l_sq res_beta*res_alpha
    fc1_sphere_weight: torch.Tensor,  # 2*C C
    fc2_sphere_weight: torch.Tensor,  # C C
    fc3_sphere_weight: torch.Tensor,  # C C
):
    x = torch.cat((x, x_message), dim=-1)
    # ^ Shape: N l_sq 2C

    # Project to the grid
    # x = torch.einsum("bai,eic->ebac", to_grid_sh_tri, x)
    # # ^ Shape: N res_beta res_alpha 2C
    x = torch.bmm(to_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
    # ^ Shape: N res_beta*res_alpha 2C

    # Conv
    x = F.silu(F.linear(x, fc1_sphere_weight))
    x = F.silu(F.linear(x, fc2_sphere_weight))
    x = F.linear(x, fc3_sphere_weight)

    # Project back to the coefficients
    # x = torch.einsum("bai,ebac->eic", from_grid_sh_tri, x)
    # # ^ Shape: N l_sq C
    x = torch.bmm(from_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
    # ^ Shape: N l_sq C

    return x


def fused_s2_pointwise_conv_unoptimized(
    x: torch.Tensor,  # N l_sq C,
    x_message: torch.Tensor,  # N l_sq C,
    to_grid_sh_tri: torch.Tensor,  # res_beta*res_alpha l_sq
    from_grid_sh_tri: torch.Tensor,  # l_sq res_beta*res_alpha
    fc1_sphere_weight: torch.Tensor,  # 2*C C
    fc2_sphere_weight: torch.Tensor,  # C C
    fc3_sphere_weight: torch.Tensor,  # C C
    masker: "Masker",
):
    with ActSave.context("s2_conv"):
        # Project to the grid
        to_grid_sh_tri = masker(to_grid_sh_tri, dim=1, with_mmax=False)
        from_grid_sh_tri = masker(from_grid_sh_tri, dim=0, with_mmax=False)
        x = torch.bmm(to_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
        ActSave({"x_grid": x})

        x_message = torch.bmm(
            to_grid_sh_tri[None].expand(x.shape[0], -1, -1), x_message
        )
        ActSave({"x_message_grid": x_message})
        # ^ Shape: N res_beta*res_alpha 2C

        x = torch.cat((x, x_message), dim=-1)
        ActSave({"x_cat": x})

        # Conv
        x = F.silu(F.linear(x, fc1_sphere_weight))
        x = F.silu(F.linear(x, fc2_sphere_weight))
        x = F.linear(x, fc3_sphere_weight)
        ActSave({"x_grid_conv": x})

        # Project back to the coefficients
        # x = torch.einsum("bai,ebac->eic", from_grid_sh_tri, x)
        # # ^ Shape: N l_sq C
        x = torch.bmm(from_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
        # ^ Shape: N l_sq C
        ActSave({"x_message": x})

        return x


@torch.no_grad()
def _grid_precomputes_optimized(
    mmax: int,
    res_beta: int,
    res_alpha: int,
    dtype: torch.dtype = torch.float,
    device: Optional[torch.device] = None,
    normalization: str = "integral",
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
) -> Float[torch.Tensor, "res_beta res_alpha m 2 l"]:
    # l_sq = (lmax+1)**2
    lmax = 2 * mmax
    assert sh.shape[-1] == (lmax + 1) ** 2
    # idx, mask = compute_idx_and_mask(mmax)
    # sh_rh = sh[..., idx] * mask

    idx, _ = compute_idx_and_mask(mmax)
    sh_rh = sh[..., idx]  # * mask
    return sh_rh


class Rhomboidal_SO3_Grid_Optimized(nn.Module):
    # Rhomboidal SO3 grid
    to_grid_sh_rh: Float[torch.Tensor, "res_beta*res_alpha m*2*l"]
    from_grid_sh_rh: Float[torch.Tensor, "res_beta res_alpha m 2 l"]

    # Triangular SO3 grid
    to_grid_sh_tri: Float[torch.Tensor, "res_beta*res_alpha l_sq"]
    from_grid_sh_tri: Float[torch.Tensor, "l_sq res_beta*res_alpha"]

    def __init__(
        self,
        mmax: int,
        res: Optional[Tuple[int, int]] = None,
        normalization: Normalization = "integral",
        grid_fp16: bool = False,
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
            functools.partial(_grid_sh_tri_to_rh, mmax=mmax),
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

        self.register_buffer("to_grid_sh_tri", to_grid_sh_tri, persistent=False)
        self.register_buffer("from_grid_sh_tri", from_grid_sh_tri, persistent=False)
        self.register_buffer("to_grid_sh_rh", to_grid_sh_rh, persistent=False)
        self.register_buffer("from_grid_sh_rh", from_grid_sh_rh, persistent=False)

    def to_s2grid_rh(
        self, coeffs: Float[torch.Tensor, "E m 2 l C"]
    ) -> Float[torch.Tensor, "E res_beta res_alpha C"]:
        # to_grid_sh_rh: Float[torch.Tensor, "res_beta res_alpha m 2 l"]
        signal = torch.einsum("bamsl,emslc->ebac", self.to_grid_sh_rh, coeffs)
        return signal

    def from_s2grid_rh(
        self, signal: Float[torch.Tensor, "E res_beta res_alpha C"]
    ) -> Float[torch.Tensor, "E m 2 l C"]:
        # from_grid_sh_rh: Float[torch.Tensor, "res_beta res_alpha m 2 l"]
        coeffs = torch.einsum("bamsl,ebac->emslc", self.from_grid_sh_rh, signal)
        return coeffs

    def to_s2grid_tri(
        self, coeffs: Float[torch.Tensor, "E l_sq C"]
    ) -> Float[torch.Tensor, "E res_beta res_alpha C"]:
        # to_grid_sh_tri: Float[torch.Tensor, "res_beta res_alpha l_sq"]
        signal = torch.einsum("bai,eic->ebac", self.to_grid_sh_tri, coeffs)
        return signal

    def from_s2grid_tri(
        self, signal: Float[torch.Tensor, "E res_beta res_alpha C"]
    ) -> Float[torch.Tensor, "E l_sq C"]:
        # from_grid_sh_tri: Float[torch.Tensor, "res_beta res_alpha l_sq"]
        coeffs = torch.einsum("bai,ebac->eic", self.from_grid_sh_tri, signal)
        return coeffs
