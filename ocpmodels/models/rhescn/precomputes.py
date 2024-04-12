from logging import getLogger
from typing import List, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import NoReturn, override

from ._jd_precomputed_data import Jd as _Jd
from .rhomboidal import compute_idx_and_mask

log = getLogger(__name__)


class WignerPrecomps(NamedTuple):
    inds: torch.Tensor
    reversed_inds: torch.Tensor
    frequencies: torch.Tensor
    mask: torch.Tensor


def _z_rot_mat_inds(
    l: int,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
):
    assert l <= lmax, f"{l=} > {lmax=}"
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    mask = torch.ones_like(inds, dtype=torch.bool, device=device)

    if l == lmax:
        return inds, reversed_inds, frequencies, mask

    # Pad the indices to the maximum l
    # use the last index as a sinkhole for the padded computation
    last_index = (2 * lmax + 1) - 1
    inds_padded = torch.full(
        (2 * lmax + 1,),
        last_index,
        device=device,
        dtype=inds.dtype,
    )
    inds_padded[: 2 * l + 1] = inds
    reversed_inds_padded = torch.full(
        (2 * lmax + 1,),
        last_index,
        device=device,
        dtype=reversed_inds.dtype,
    )
    reversed_inds_padded[: 2 * l + 1] = reversed_inds
    frequencies_padded = torch.full(
        (2 * lmax + 1,),
        0.0,
        device=device,
        dtype=frequencies.dtype,
    )
    frequencies_padded[: 2 * l + 1] = frequencies

    mask_padded = torch.zeros((2 * lmax + 1,), dtype=torch.bool, device=device)
    mask_padded[: 2 * l + 1] = mask

    return inds_padded, reversed_inds_padded, frequencies_padded, mask_padded


def _z_rot_mat_parallel_precomps(lmax: int, dtype: torch.dtype, device: torch.device):
    ls = list(range(lmax + 1))
    all_inds = []
    all_reversed_inds = []
    all_frequencies = []
    all_masks = []
    for l in ls:
        inds, reversed_inds, frequencies, mask = _z_rot_mat_inds(
            l, max(ls), device, dtype
        )

        all_inds.append(inds)
        all_reversed_inds.append(reversed_inds)
        all_frequencies.append(frequencies)
        all_masks.append(mask)

    inds = torch.stack(all_inds, dim=0)
    reversed_inds = torch.stack(all_reversed_inds, dim=0)
    frequencies = torch.stack(all_frequencies, dim=0)
    mask = torch.stack(all_masks, dim=0)
    return WignerPrecomps(inds, reversed_inds, frequencies, mask)


class JdPrecomputed(nn.Module):
    Jd_padded: torch.Tensor

    precomps_inds: torch.Tensor
    precomps_reversed_inds: torch.Tensor
    precomps_frequencies: torch.Tensor
    precomps_mask: torch.Tensor

    def __init__(
        self,
        lmax: int,
        wigner_fp16: bool = False,
        store_individually: bool = False,
    ):
        super().__init__()

        self.wigner_fp16 = wigner_fp16

        padded_list: List[torch.Tensor] = []
        for l in range(lmax + 1):
            Jd_l = _Jd[l]  # 2l+1 2l+1

            if store_individually:
                self.register_buffer(f"Jd_{l}", Jd_l, persistent=False)

            # Pad it so that it's 2lmax+1 2lmax+1
            Jd_l_padded = F.pad(
                Jd_l,
                (0, 2 * lmax - 2 * l, 0, 2 * lmax - 2 * l),
                mode="constant",
                value=0,
            )  # 2lmax+1 2lmax+1
            padded_list.append(Jd_l_padded)

        Jd_padded = torch.stack(padded_list, dim=0)  # lmax+1 2lmax+1 2lmax+1
        if wigner_fp16:
            Jd_padded = Jd_padded.half()
        self.register_buffer("Jd_padded", Jd_padded, persistent=False)

        # compute the precomps
        precomps = _z_rot_mat_parallel_precomps(lmax, Jd_padded.dtype, Jd_padded.device)
        self.register_buffer("precomps_inds", precomps.inds, persistent=False)
        self.register_buffer(
            "precomps_reversed_inds", precomps.reversed_inds, persistent=False
        )
        self.register_buffer(
            "precomps_frequencies",
            precomps.frequencies.half() if wigner_fp16 else precomps.frequencies,
            persistent=False,
        )
        self.register_buffer("precomps_mask", precomps.mask, persistent=False)

    @property
    def wp(self):
        return WignerPrecomps(
            self.precomps_inds,
            self.precomps_reversed_inds,
            self.precomps_frequencies,
            self.precomps_mask,
        )

    @override
    def forward(self) -> NoReturn:
        raise ValueError("This module is not callable.")


class RhomboidalPrecomputes(nn.Module):
    rh_idx: torch.Tensor
    rh_mask: torch.Tensor

    @override
    def __init__(
        self,
        lmax: int,
        mmax: int,
        wigner_fp16: bool = False,
    ) -> None:
        super().__init__()

        # Set up rhomboidal idx/mask
        idx, mask = compute_idx_and_mask(mmax)
        self.register_buffer("rh_idx", idx, persistent=False)
        self.register_buffer("rh_mask", mask, persistent=False)

        self.jdp = JdPrecomputed(lmax, wigner_fp16=wigner_fp16)

    @override
    def forward(self) -> NoReturn:
        raise ValueError("This module is not callable.")
