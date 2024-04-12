from logging import getLogger
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from einops import rearrange, repeat
from jaxtyping import Float
from ll import ActSave
from typing_extensions import override

from ocpmodels.modules import profiler as P

from .jd_precomputed import Jd as _Jd

if TYPE_CHECKING:
    from .model import Masker

log = getLogger(__name__)


def _init_edge_rot_mat(edge_distance_vec, seed: int | None = None):
    generator = torch.Generator(device=edge_distance_vec.device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    # if torch.min(edge_vec_0_distance) < 0.0001:
    #     log.error(
    #         "Error edge_vec_0_distance: {}".format(
    #             torch.min(edge_vec_0_distance)
    #         )
    #     )
    #     (minval, minidx) = torch.min(edge_vec_0_distance, 0)
    #     log.error("Error edge_vec_0_distance: {}".format(minidx))

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = (
        torch.randn(
            edge_vec_0.shape,
            generator=generator,
            device=edge_vec_0.device,
            dtype=edge_vec_0.dtype,
        )
        - 0.5
    )
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    # assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()


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


class JdPrecomputed_Optimized(nn.Module):
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
    def forward(self) -> Float[torch.Tensor, "lmax+1 2lmax+1 2lmax+1"]:
        return self.Jd_padded


# @dataclass(frozen=True)
class WignerPrecomps(NamedTuple):
    inds: torch.Tensor
    reversed_inds: torch.Tensor
    frequencies: torch.Tensor
    mask: torch.Tensor


def _z_rot_mat_parallel(
    angle: torch.Tensor,  # (N,)
    lmax: int,
    inds: torch.Tensor,  # (lmax + 1, 2 * lmax + 1)
    reversed_inds: torch.Tensor,  # (lmax + 1, 2 * lmax + 1)
    frequencies: torch.Tensor,  # (lmax + 1, 2 * lmax + 1)
    mask: torch.Tensor,  # (lmax + 1, 2 * lmax + 1)
    device: torch.device,
):
    assert inds.shape == (lmax + 1, 2 * lmax + 1)
    assert reversed_inds.shape == (lmax + 1, 2 * lmax + 1)
    assert frequencies.shape == (lmax + 1, 2 * lmax + 1)
    assert angle.ndim == 1

    # Output shape: (N, lmax + 1, 2 * lmax + 1, 2 * lmax + 1)
    sin = torch.sin(frequencies * angle[..., None, None])  # (N, lmax + 1, 2 * lmax + 1)
    cos = torch.cos(frequencies * angle[..., None, None])  # (N, lmax + 1, 2 * lmax + 1

    M = sin.new_zeros(
        (*angle.shape, lmax + 1, 2 * lmax + 1, 2 * lmax + 1),
        device=device,
    )

    N_range = torch.arange(angle.shape[0], device=device)[:, None, None]
    lmax_range = torch.arange(lmax + 1, device=device)[None, :, None]
    M[N_range, lmax_range, inds[None, :, :], reversed_inds[None, :, :]] = sin
    M[N_range, lmax_range, inds[None, :, :], inds[None, :, :]] = cos

    # M: (N, lmax + 1, 2 * lmax + 1, 2 * lmax + 1)
    # mask: (lmax + 1, 2 * lmax + 1)
    M = M * mask[None, :, :, None]

    return M


def _z_rot_mat_parallel_vectorized(
    angle: torch.Tensor,  # Float[torch.Tensor, "a N"],
    lmax: int,
    inds: torch.Tensor,  # Float[torch.Tensor, "lmax+1 2*lmax+1"],
    reversed_inds: torch.Tensor,  # Float[torch.Tensor, "lmax+1 2*lmax+1"],
    frequencies: torch.Tensor,  # Float[torch.Tensor, "lmax+1 2*lmax+1"],
    mask: torch.Tensor,  # Float[torch.Tensor, "lmax+1 2*lmax+1"],
    device: torch.device,
):
    assert inds.shape == (lmax + 1, 2 * lmax + 1)
    assert reversed_inds.shape == (lmax + 1, 2 * lmax + 1)
    assert frequencies.shape == (lmax + 1, 2 * lmax + 1)
    assert angle.ndim == 2

    # Output shape: (N, lmax + 1, 2 * lmax + 1, 2 * lmax + 1)
    sin = torch.sin(
        frequencies * angle[..., None, None]
    )  # (a, N, lmax + 1, 2 * lmax + 1)
    cos = torch.cos(
        frequencies * angle[..., None, None]
    )  # (a, N, lmax + 1, 2 * lmax + 1)

    M = sin.new_zeros(
        (angle.shape[0], angle.shape[1], lmax + 1, 2 * lmax + 1, 2 * lmax + 1),
        device=device,
    )

    a_range = torch.arange(angle.shape[0], device=device)[:, None, None, None]
    N_range = torch.arange(angle.shape[1], device=device)[None, :, None, None]
    lmax_range = torch.arange(lmax + 1, device=device)[None, None, :, None]
    inds = inds[None, None, :, :]
    reversed_inds = reversed_inds[None, None, :, :]
    M[a_range, N_range, lmax_range, inds, reversed_inds] = sin
    M[a_range, N_range, lmax_range, inds, inds] = cos

    # M: (N, lmax + 1, 2 * lmax + 1, 2 * lmax + 1)
    # mask: (lmax + 1, 2 * lmax + 1)
    M = M * mask[None, None, :, :, None]

    return M


def direct_sum_dense(
    dense: torch.Tensor,  # Float[torch.Tensor, "E lmax+1 2*lmax+1 2*lmax+1"],
    lmax: int,
    # ) -> Float[torch.Tensor, "E l_sq l_sq"]:
) -> torch.Tensor:
    assert dense.shape[1:] == (lmax + 1, 2 * lmax + 1, 2 * lmax + 1)

    out = torch.zeros(
        (dense.shape[0], int((lmax + 1) ** 2), int((lmax + 1) ** 2)),
        dtype=dense.dtype,
        device=dense.device,
    )
    i, j = 0, 0
    for l in range(lmax + 1):
        wigner = dense[:, l]  # N (2*lmax+1) (2*lmax+1)
        wigner = wigner[..., : 2 * l + 1, : 2 * l + 1]  # N (2*l+1) (2*l+1)

        m, n = wigner.shape[-2:]
        out[..., i : i + m, j : j + n] = wigner
        j += n
        i += m

    return out


def wigner_from_xyz_parallel(
    lmax: int,
    xyz: Float[torch.Tensor, "E 3"],
    jdp: JdPrecomputed_Optimized,
    wp: WignerPrecomps,
    vectorize: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Float[torch.Tensor, "E l_sq l_sq"]:
    if device is None:
        device = xyz.device
    if dtype is None:
        dtype = xyz.dtype

    alpha, beta = o3.xyz_to_angles(xyz)

    if not vectorize:
        # ^ Shape: (lmax+1) (2*lmax+1) (2*lmax+1)
        Xa = _z_rot_mat_parallel(alpha, lmax, *wp, device)
        # ^ Shape: N (lmax+1) (2*lmax+1) (2*lmax+1)
        Xb = _z_rot_mat_parallel(beta, lmax, *wp, device)
        # ^ Shape: N (lmax+1) (2*lmax+1) (2*lmax+1)
    else:
        with P.record_function("z_rot_mat_parallel_vectorized"):
            a = torch.stack([alpha, beta], dim=0)
            X = _z_rot_mat_parallel_vectorized(a, lmax, *wp, device)
            Xa, Xb = X.unbind(0)

    J = jdp.Jd_padded
    wigner_dense: Float[torch.Tensor, "E lmax+1 2*lmax+1 2*lmax+1"] = Xa @ J @ Xb @ J
    wigner = direct_sum_dense(wigner_dense, lmax)
    return wigner


def _xyz_to_angles(xyz: torch.Tensor):
    xyz = F.normalize(xyz, p=2.0, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


@torch.jit.script
def wigner_matmul(
    Xa: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    Xb: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    Xc: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    J: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
) -> torch.Tensor:
    # return Xa @ J @ Xb @ J @ Xc
    return torch.bmm(torch.bmm(torch.bmm(torch.bmm(Xa, J), Xb), J), Xc)


# @torch.jit.script
def wigner_from_rot_mat_parallel(
    lmax: int,
    edge_rot_mat: torch.Tensor,  # Float[torch.Tensor, "E 3 3"],
    J: torch.Tensor,
    inds: torch.Tensor,
    reversed_inds: torch.Tensor,
    frequencies: torch.Tensor,
    mask: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    # ) -> Float[torch.Tensor, "E l_sq l_sq"]:
) -> torch.Tensor:
    if device is None:
        device = J.device
    if dtype is None:
        dtype = J.dtype

    # x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
    with P.record_function("compute_alpha_beta_gamma"):
        # x = edge_rot_mat @ torch.tensor(
        #     [0.0, 1.0, 0.0],
        #     device=edge_rot_mat.device,
        #     dtype=edge_rot_mat.dtype,
        # )
        # ^ is the same as:
        x = edge_rot_mat[:, :, 1]
        alpha, beta = _xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

    a = torch.stack([alpha, beta, gamma], dim=0)
    with P.record_function("z_rot_mat_parallel_vectorized"):
        X = _z_rot_mat_parallel_vectorized(
            a,
            lmax,
            inds,
            reversed_inds,
            frequencies,
            mask,
            device,
        )
    E = X.shape[1]
    X = X.view(X.shape[0], -1, X.shape[-2], X.shape[-1])
    Xa, Xb, Xc = X.unbind(0)

    with P.record_function("repeat_Jp"):
        J = repeat(J, "lmax twol1 twol2 -> (E lmax) twol1 twol2", E=E)

    with P.record_function("wigner_matmul"):
        with torch.autocast(J.device.type, enabled=False):
            # wigner_dense = Xa @ J @ Xb @ J @ Xc
            wigner_dense = wigner_matmul(Xa, Xb, Xc, J)
            wigner_dense = wigner_dense.view(
                E, -1, wigner_dense.shape[-2], wigner_dense.shape[-1]
            )

    wigner = direct_sum_dense(wigner_dense, lmax)
    return wigner


class Rhomboidal_SO3_Rotation_Parallel_FastMM(NamedTuple):
    wigner: Float[torch.Tensor, "E m*2*m l_sq"]
    wigner_inv: Float[torch.Tensor, "E m1*2*m2 l_sq"] | None
    wigner_inv_from_grid: Float[torch.Tensor, "E l_sq res_beta*res_alpha"]
    full_wigner: Float[torch.Tensor, "E l_sq l_sq"] | None
    full_wigner_inv: Float[torch.Tensor, "E l_sq l_sq"] | None
    rotmat: Float[torch.Tensor, "E 3 3"] | None

    # rh_idx: torch.Tensor | None = None
    # rh_mask: torch.Tensor | None = None


def rot_from_xyz(
    xyz: Float[torch.Tensor, "E 3"],
    from_grid_sh_tri: Float[torch.Tensor, "res_beta*res_alpha l_sq"],
    jdp: JdPrecomputed_Optimized,
    mmax: int,
    rh_idx: torch.Tensor,
    # rh_mask: torch.Tensor,
    use_rotmat: bool = True,
    xyz_is_rotmat: bool = False,
    keep_full_wigner: bool = False,
    keep_full_wigner_inv: bool = False,
    keep_rot_mat: bool = False,
    rotmat_seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    vectorize: bool = False,
    *,
    masker: "Masker",
):
    device = xyz.device if device is None else device
    dtype = xyz.dtype if dtype is None else dtype

    lmax = mmax * 2
    if use_rotmat:
        if xyz_is_rotmat:
            rot_mat = xyz
        else:
            with P.record_function("init_edge_rot_mat"):
                rot_mat = _init_edge_rot_mat(xyz, rotmat_seed)
                if jdp.wigner_fp16:
                    rot_mat = rot_mat.half()
        with torch.autocast(device.type, enabled=False):
            with P.record_function("wigner_from_rot_mat_parallel"):
                full_wigner = wigner_from_rot_mat_parallel(
                    lmax,
                    rot_mat,
                    jdp.Jd_padded,
                    *jdp.wp,
                )
    else:
        full_wigner = wigner_from_xyz_parallel(
            lmax,
            xyz,
            jdp,
            jdp.wp,
            vectorize=vectorize,
        )
        rot_mat = None

    ActSave({"full_wigner": full_wigner})

    with P.record_function("prepare_wigner"):
        wigner = full_wigner.transpose(-1, -2)[..., rh_idx]  # * rh_mask[None, None]

        # wigner = repeat(wigner, "... -> two_st ...", two_st=2)
        wigner = rearrange(wigner, "E l_sq m1 two_sign m2 -> E (m1 two_sign m2) l_sq")

        if jdp.wigner_fp16:
            wigner = wigner.half()

        wigner = wigner.contiguous()
        ActSave({"wigner_tri_to_rh": wigner})

    with P.record_function("prepare_wigner_inv"):
        wigner_inv: Float[torch.Tensor, "E m1*2*m2 l_sq"] | None = None
        if False:
            wigner_inv = (
                full_wigner.transpose(-1, -2)[:, rh_idx]  # * rh_mask[None, ..., None]
            )

            wigner_inv = rearrange(
                wigner_inv,
                "E m1 two_sign m2 l_sq -> E (m1 two_sign m2) l_sq",
            )

    with P.record_function("prepare_wigner_inv_from_grid"):
        # wigner_inv_from_grid = torch.einsum(
        #     "eij,ig->ejg", full_wigner, from_grid_sh_tri
        # )

        wigner_inv_full = full_wigner.transpose(-1, -2)
        wigner_inv_full = masker(wigner_inv_full, dim=1, with_mmax=False)
        wigner_inv_full = masker(wigner_inv_full, dim=2, with_mmax=True)

        wigner_inv_from_grid = torch.bmm(
            wigner_inv_full,
            from_grid_sh_tri[None].expand(full_wigner.shape[0], -1, -1),
        )

        if jdp.wigner_fp16:
            wigner_inv_from_grid = wigner_inv_from_grid.half()

        wigner_inv_from_grid = wigner_inv_from_grid.contiguous()
        ActSave({"wigner_inv_from_grid": wigner_inv_from_grid})

    return Rhomboidal_SO3_Rotation_Parallel_FastMM(
        wigner,
        wigner_inv,
        wigner_inv_from_grid,
        full_wigner=full_wigner if keep_full_wigner else None,
        full_wigner_inv=wigner_inv_full if keep_full_wigner_inv else None,
        rotmat=rot_mat if keep_rot_mat else None,
        # rh_idx=rh_idx,
        # rh_mask=rh_mask,
    )


def rot_from_xyz_new(
    xyz: Float[torch.Tensor, "E 3"],
    from_grid_sh_tri: Float[torch.Tensor, "res_beta*res_alpha l_sq"],
    jdp: JdPrecomputed_Optimized,
    mmax: int,
    rh_idx: torch.Tensor,
    rh_mask: torch.Tensor,
    keep_full_wigner: bool = False,
    keep_full_wigner_inv: bool = False,
    keep_rot_mat: bool = False,
    rotmat_seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    device = xyz.device if device is None else device
    dtype = xyz.dtype if dtype is None else dtype

    xyz = F.normalize(xyz, p=2.0, dim=-1)  # forward 0's instead of nan for zero-radius

    lmax = mmax * 2
    with P.record_function("init_edge_rot_mat"):
        rot_mat = _init_edge_rot_mat(xyz, rotmat_seed)
        rot_mat_rev = _init_edge_rot_mat(-xyz, rotmat_seed)

        if jdp.wigner_fp16:
            rot_mat = rot_mat.half()
            rot_mat_rev = rot_mat_rev.half()

    with torch.autocast(device.type, enabled=False):
        with P.record_function("wigner_from_rot_mat_parallel"):
            full_wigner = wigner_from_rot_mat_parallel(
                lmax,
                rot_mat,
                jdp.Jd_padded,
                *jdp.wp,
            )
            full_wigner_rev = wigner_from_rot_mat_parallel(
                lmax,
                rot_mat_rev,
                jdp.Jd_padded,
                *jdp.wp,
            )

    with P.record_function("prepare_wigner"):
        wigner = full_wigner[..., rh_idx]  # * rh_mask[None, None]
        # wigner_rev = full_wigner_rev[..., rh_idx]  # * rh_mask[None, None]
        wigner_rev = full_wigner.transpose(-1, -2)[..., rh_idx]  # * rh_mask[None, None]

        # wigner = repeat(wigner, "... -> two_st ...", two_st=2)
        wigner = torch.stack([wigner, wigner_rev], dim=0)
        wigner = rearrange(
            wigner,
            "two_st E l_sq m1 two_sign m2 -> (two_st E) (m1 two_sign m2) l_sq",
        )

        if jdp.wigner_fp16:
            wigner = wigner.half()

        wigner = wigner.contiguous()

    with P.record_function("prepare_wigner_inv"):
        wigner_inv: Float[torch.Tensor, "E m1*2*m2 l_sq"] | None = None
        if False:
            wigner_inv = (
                full_wigner.transpose(-1, -2)[:, rh_idx] * rh_mask[None, ..., None]
            )

            wigner_inv = rearrange(
                wigner_inv,
                "E m1 two_sign m2 l_sq -> E (m1 two_sign m2) l_sq",
            )

    with P.record_function("prepare_wigner_inv_from_grid"):
        # wigner_inv_from_grid = torch.einsum(
        #     "eij,ig->ejg", full_wigner, from_grid_sh_tri
        # )
        wigner_inv_from_grid = torch.bmm(
            full_wigner.transpose(-1, -2),
            from_grid_sh_tri[None].expand(full_wigner.shape[0], -1, -1),
        )

        if jdp.wigner_fp16:
            wigner_inv_from_grid = wigner_inv_from_grid.half()

        wigner_inv_from_grid = wigner_inv_from_grid.contiguous()

    return Rhomboidal_SO3_Rotation_Parallel_FastMM(
        wigner,
        wigner_inv,
        wigner_inv_from_grid,
        full_wigner=full_wigner if keep_full_wigner else None,
        full_wigner_inv=full_wigner.transpose(-1, -2) if keep_full_wigner_inv else None,
        rotmat=rot_mat if keep_rot_mat else None,
        rh_idx=rh_idx,
        rh_mask=rh_mask,
    )
