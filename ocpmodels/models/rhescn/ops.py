from logging import getLogger

import torch
import torch.nn.functional as F

log = getLogger(__name__)


@torch.jit.script
def fused_s2_pointwise_nonlinearity_and_rotate_inv(
    x: torch.Tensor,  # E (m 2 l) C
    to_grid_sh_rh: torch.Tensor,  # (res_beta res_alpha) (m 2 l)
    wigner_inv_from_grid: torch.Tensor,  # E (res_beta res_alpha) l_sq
):
    x = torch.bmm(to_grid_sh_rh[None].expand(x.shape[0], -1, -1), x)
    x = F.silu(x)
    x = torch.bmm(wigner_inv_from_grid, x)
    return x


@torch.jit.script
def fused_s2_pointwise_conv(
    x: torch.Tensor,  # N l_sq C,
    x_message: torch.Tensor,  # N l_sq C,
    to_grid_sh_tri: torch.Tensor,  # res_beta*res_alpha l_sq
    from_grid_sh_tri: torch.Tensor,  # l_sq res_beta*res_alpha
    fc1_sphere_weight: torch.Tensor,  # 2*C C
    fc2_sphere_weight: torch.Tensor,  # C C
    fc3_sphere_weight: torch.Tensor,  # C C
):
    # Project to the grid
    x = torch.cat((x, x_message), dim=-1)
    # ^ Shape: N l_sq 2C

    x = torch.bmm(to_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
    # ^ Shape: N res_beta*res_alpha 2C

    # Conv
    x = F.silu(F.linear(x, fc1_sphere_weight))
    x = F.silu(F.linear(x, fc2_sphere_weight))
    x = F.linear(x, fc3_sphere_weight)

    x = torch.bmm(from_grid_sh_tri[None].expand(x.shape[0], -1, -1), x)
    # ^ Shape: N l_sq C

    return x
