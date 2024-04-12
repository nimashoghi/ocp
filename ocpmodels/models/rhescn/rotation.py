from logging import getLogger
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from e3nn import o3
from einops import rearrange, repeat
from jaxtyping import Float
from ll import ActSave

from ocpmodels.modules import profiler as P

from .escn_compat import escn_compat_apply_tri_mask, eSCNCompatConfig
from .precomputes import RhomboidalPrecomputes

log = getLogger(__name__)


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


def _direct_sum_dense(
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


def _xyz_to_angles(xyz: torch.Tensor):
    xyz = F.normalize(xyz, p=2.0, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


@torch.jit.script
def _wigner_matmul(
    Xa: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    Xb: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    Xc: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
    J: torch.Tensor,  # (E*(lmax+1)) (2*lmax+1) (2*lmax+1)
) -> torch.Tensor:
    # return Xa @ J @ Xb @ J @ Xc
    return torch.bmm(torch.bmm(torch.bmm(torch.bmm(Xa, J), Xb), J), Xc)


# @torch.jit.script
def _wigner_from_rot_mat_parallel(
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
            wigner_dense = _wigner_matmul(Xa, Xb, Xc, J)
            wigner_dense = wigner_dense.view(
                E, -1, wigner_dense.shape[-2], wigner_dense.shape[-1]
            )

    wigner = _direct_sum_dense(wigner_dense, lmax)
    return wigner


class ComputedWignerMatrices(NamedTuple):
    wigner: Float[torch.Tensor, "E m*2*m l_sq"]
    wigner_inv_from_grid: Float[torch.Tensor, "E l_sq res_beta*res_alpha"]


def _init_edge_rot_mat(edge_distance_vec: torch.Tensor, seed: int | None = None):
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


def rot_from_xyz(
    xyz: Float[torch.Tensor, "E 3"],
    *,
    mmax: int,
    from_grid_sh_tri: Float[torch.Tensor, "res_beta*res_alpha l_sq"],
    precomputes: RhomboidalPrecomputes,
    rh_idx: torch.Tensor,
    rotmat_seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    escn_compat_config: eSCNCompatConfig | None,
):
    device = xyz.device if device is None else device
    dtype = xyz.dtype if dtype is None else dtype

    lmax = mmax * 2
    jdp = precomputes.jdp

    with P.record_function("init_edge_rot_mat"):
        rot_mat = _init_edge_rot_mat(xyz, rotmat_seed)
        if jdp.wigner_fp16:
            rot_mat = rot_mat.half()
    with torch.autocast(device.type, enabled=False):
        with P.record_function("wigner_from_rot_mat_parallel"):
            full_wigner = _wigner_from_rot_mat_parallel(
                lmax,
                rot_mat,
                jdp.Jd_padded,
                *jdp.wp,
            )

    ActSave({"full_wigner": full_wigner})

    with P.record_function("prepare_wigner"):
        wigner = full_wigner.transpose(-1, -2)[..., rh_idx]  # * rh_mask[None, None]

        # wigner = repeat(wigner, "... -> two_st ...", two_st=2)
        wigner = rearrange(wigner, "E l_sq m1 two_sign m2 -> E (m1 two_sign m2) l_sq")

        if jdp.wigner_fp16:
            wigner = wigner.half()

        wigner = wigner.contiguous()
        ActSave({"wigner_tri_to_rh": wigner})

    with P.record_function("prepare_wigner_inv_from_grid"):
        # wigner_inv_from_grid = torch.einsum(
        #     "eij,ig->ejg", full_wigner, from_grid_sh_tri
        # )

        wigner_inv_full = full_wigner.transpose(-1, -2)
        wigner_inv_full = escn_compat_apply_tri_mask(
            escn_compat_config,
            wigner_inv_full,
            dim=1,
            with_mmax=False,
        )
        wigner_inv_full = escn_compat_apply_tri_mask(
            escn_compat_config,
            wigner_inv_full,
            dim=2,
            with_mmax=True,
        )

        wigner_inv_from_grid = torch.bmm(
            wigner_inv_full,
            from_grid_sh_tri[None].expand(full_wigner.shape[0], -1, -1),
        )

        if jdp.wigner_fp16:
            wigner_inv_from_grid = wigner_inv_from_grid.half()

        wigner_inv_from_grid = wigner_inv_from_grid.contiguous()
        ActSave({"wigner_inv_from_grid": wigner_inv_from_grid})

    return ComputedWignerMatrices(wigner, wigner_inv_from_grid)
