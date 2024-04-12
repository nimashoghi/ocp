"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import time
from typing import List

import lovely_tensors as lt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from jaxtyping import Float, Int
from ll import ActSave
from torch.autograd import profiler as P
from torch_scatter import scatter
from typing_extensions import override

from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from ..rhescnv10.util import tassert
from .ckptutil import filter_ckpt
from .radius_graph_utils import generate_graph
from .rhomboidal import compute_idx_and_mask, signidx
from .so3 import (
    Rhomboidal_SO3_Grid_Optimized,
    fused_s2_pointwise_conv_optimized,
    fused_s2_pointwise_conv_unoptimized,
    fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_optimized,
    fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_unoptimized,
)
from .sphharm import spherical_harmonics
from .util import rearrange_view
from .wigner_efficient import (
    JdPrecomputed_Optimized,
    Rhomboidal_SO3_Rotation_Parallel_FastMM,
    rot_from_xyz,
)

lt.monkey_patch()
# from ocpmodels.modules.profiler import active_profiler


GridType = Rhomboidal_SO3_Grid_Optimized
JdpType = JdPrecomputed_Optimized
RotType = Rhomboidal_SO3_Rotation_Parallel_FastMM


OPTIMIZE = True

try:
    from e3nn import o3
except ImportError:
    pass


class Masker:
    def __init__(
        self,
        lmax: int | None,
        mmax: int | None,
    ):
        self.lmax = lmax
        self.mmax = mmax

    def rh(
        self,
        x: torch.Tensor,
        dim: int = 1,
        reshape: bool = True,
    ):
        if reshape:
            m = int((x.shape[dim] // 2) ** 0.5)
            x = rearrange_view(
                x, "E (m two_sign l) C -> E m two_sign l C", m=m, two_sign=2, l=m
            )
        if self.mmax is None or self.mmax > x.shape[dim]:
            if reshape:
                x = rearrange_view(x, "E m two_sign l C -> E (m two_sign l) C")
            return x

        idx = ()
        for i in range(dim):
            idx += (slice(None),)

        idx_to_keep = idx + (slice(self.mmax),)
        idx_to_zero = idx + (slice(self.mmax, None),)

        x = torch.cat([x[idx_to_keep], torch.zeros_like(x[idx_to_zero])], dim=dim)
        if reshape:
            x = rearrange_view(x, "E m two_sign l C -> E (m two_sign l) C")
        return x

    def __call__(
        self,
        x: torch.Tensor,
        dim: int = 1,
        with_mmax: bool = False,
    ):
        if self.lmax is None:
            return x
        count = (self.lmax + 1) ** 2

        idx = ()
        for i in range(dim):
            idx += (slice(None),)

        idx_to_keep = idx + (slice(count),)
        idx_to_zero = idx + (slice(count, None),)

        x = torch.cat([x[idx_to_keep], torch.zeros_like(x[idx_to_zero])], dim=dim)

        if with_mmax:
            x = self.tri_mmax(x, dim=dim)
        return x

    def tri_mmax(self, x: torch.Tensor, dim: int = 1):
        if self.mmax is None:
            return x

        # X is shape (E, (lmax+1)^2, C)
        # Let's get the lmax
        lmax = int((x.shape[dim]) ** 0.5) - 1

        i = 0
        mask = torch.ones((x.shape[dim]), dtype=torch.bool, device=x.device)
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                if abs(m) > self.mmax:
                    mask[i] = False
                i += 1

        #  Add Nones to the mask to make it the right shape
        for _ in range(dim):
            mask = mask.unsqueeze(0)
        for _ in range(dim + 1, x.dim()):
            mask = mask.unsqueeze(-1)

        x = x * mask
        return x


class M0L0Embedding(nn.Module):
    @override
    def __init__(
        self,
        num_atoms: int,
        sphere_channels: int,
        lmax: int,
    ):
        super().__init__()

        self.lmax = lmax
        self.embedding = nn.Embedding(num_atoms, sphere_channels)

    @override
    def forward(self, atomic_numbers: Float[torch.Tensor, "N"]):
        with P.record_function("M0L0Embedding"), ActSave.context("M0L0Embedding"):
            x_l0m0 = self.embedding(atomic_numbers)  # N C
            ActSave({"x_l0m0": x_l0m0})
            x_l0m0 = x_l0m0[..., None, :]  # N C 1
            x = torch.cat(
                (
                    x_l0m0,
                    torch.zeros(
                        x_l0m0.shape[0],
                        (self.lmax + 1) ** 2 - x_l0m0.shape[1],
                        x_l0m0.shape[2],
                        device=x_l0m0.device,
                        dtype=x_l0m0.dtype,
                    ),  # N C (lmax+1)^2-1
                ),
                dim=1,
            )  # N C (lmax+1)^2
            ActSave({"x": x})
            return x


def _opt_einsum():
    try:
        import torch.backends.opt_einsum

        _ = torch.backends.opt_einsum.set_flags(True, "optimal")
        # _ = torch.backends.opt_einsum.set_flags(False)
    except ImportError:
        pass


@registry.register_model("rhescnv8clean")
class RHESCN(BaseModel):
    sphere_points: Float[torch.Tensor, "P 3"]
    sphharm_weights: Float[torch.Tensor, "P l_sq"]

    rh_idx: torch.Tensor
    # rh_mask: torch.Tensor

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_neighbors: int = 40,
        cutoff: float = 8.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        lmax_list: List[int] = [6],
        mmax_list: List[int] = [2],
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        use_grid: bool = True,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
        wigner_fp16: bool = False,
        grid_fp16: bool = False,
        opt_einsum: bool = True,
        opt_wigner: bool = True,
        grid_res: tuple[int, int] | None = None,
        conv_grid_res: tuple[int, int] | None = None,
        load_from_ckpt: str | None = None,
        x_message_lmax: int | None = None,
        x_message_mmax: int | None = None,
    ) -> None:
        super().__init__()

        if opt_einsum:
            _opt_einsum()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install the e3nn library to use the SCN model")
            raise ImportError

        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_atoms = 0
        self.sphere_channels = sphere_channels
        self.max_neighbors = max_neighbors
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function
        self.wigner_fp16 = wigner_fp16
        self.opt_wigner = opt_wigner

        self.jdp = JdpType(self.lmax_list[0], wigner_fp16=self.wigner_fp16)

        assert self.num_resolutions == 1, "Only one resolution is currently supported"
        assert len(self.mmax_list) == 1, "Only one resolution is currently supported"
        assert len(self.lmax_list) == 1, "Only one resolution is currently supported"
        assert (
            self.lmax_list[0] == self.mmax_list[0] * 2
        ), f"Lmax must be twice the mmax. Got lmax: {self.lmax_list}, mmax: {self.mmax_list}"

        # variables used for display purposes
        # self.counter = 0

        # Set up rhomboidal idx/mask
        # idx, mask = compute_idx_and_mask(self.mmax_list[0])
        # self.register_buffer("rh_idx", idx, persistent=False)
        # self.register_buffer("rh_mask", mask, persistent=False)
        idx, _ = compute_idx_and_mask(self.mmax_list[0])
        self.register_buffer("rh_idx", idx, persistent=False)
        # self.register_buffer("rh_mask", mask, persistent=False)

        # non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = M0L0Embedding(
            self.max_num_elements, self.sphere_channels, self.lmax_list[0]
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]
        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        # Initialize the transformations between spherical and grid representations
        # self.SO3_grid = nn.ModuleList()
        # for lval in range(max(self.lmax_list) + 1):
        #     SO3_m_grid = nn.ModuleList()
        #     for m in range(max(self.lmax_list) + 1):
        #         SO3_m_grid.append(SO3_Grid(lval, m))

        #     self.SO3_grid.append(SO3_m_grid)

        # self.SO3_grid = SO3_Grid(self.lmax_list[0], self.lmax_list[0])
        self.rh_grid = GridType(self.mmax_list[0], grid_fp16=grid_fp16, res=grid_res)

        if conv_grid_res is not None:
            conv_rh_grid = GridType(
                self.mmax_list[0], grid_fp16=grid_fp16, res=conv_grid_res
            )
        else:
            conv_rh_grid = self.rh_grid

        self.masker = Masker(x_message_lmax, x_message_mmax)

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.distance_expansion,
                self.max_num_elements,
                # self.SO3_grid,
                self.rh_grid,
                self.act,
                conv_rh_grid=conv_rh_grid,
                masker=self.masker,
            )
            self.layer_blocks.append(block)

        # Output blocks for energy and forces
        self.energy_block = EnergyBlock(self.sphere_channels_all, self.act)
        if self.regress_forces:
            self.force_block = ForceBlock(self.sphere_channels_all, self.act)

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        sphere_points = CalcSpherePoints(num_sphere_samples)
        self.register_buffer("sphere_points", sphere_points, persistent=False)

        # For each spherical point, compute the spherical harmonic coefficient weights
        sphharm_weights = spherical_harmonics(
            torch.arange(0, self.lmax_list[0] + 1).tolist(),
            sphere_points,
            False,
        )
        self.register_buffer("sphharm_weights", sphharm_weights, persistent=False)

        if load_from_ckpt:
            ckpt = torch.load(load_from_ckpt, map_location="cpu")
            self.load_from_escn_state_dict(
                filter_ckpt(ckpt["state_dict"], "module.module.")
            )

    def load_from_escn_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.sphere_embedding.embedding.load_state_dict(
            filter_ckpt(state_dict, "sphere_embedding.")
        )
        self.distance_expansion.load_state_dict(
            filter_ckpt(state_dict, "distance_expansion.")
        )
        self.energy_block.load_from_escn_state_dict(
            filter_ckpt(state_dict, "energy_block.")
        )
        self.force_block.load_from_escn_state_dict(
            filter_ckpt(state_dict, "force_block.")
        )

        for i, block in enumerate(self.layer_blocks):
            block.load_from_escn_state_dict(
                filter_ckpt(state_dict, f"layer_blocks.{i}.")
            )

    # @conditional_grad(torch.enable_grad())
    def forward(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        # cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_distance_vec: torch.Tensor,
        batch: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # def forward(self, batch):
        if False:
            pos = batch.pos
            atomic_numbers = batch.atomic_numbers
            cell = batch.cell
            natoms = batch.natoms
            # edge_index = batch.edge_index
            # edge_distance = batch.edge_distance
            # edge_distance_vec = batch.edge_distance_vec
            batch = batch.batch

            (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                _,  # cell offset distances
                neighbors,
            ) = generate_graph(
                pos,
                cell,
                batch,
                natoms,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_pbc=self.use_pbc,
                otf_graph=self.otf_graph,
            )

        device = pos.device
        self.batch_size = natoms.shape[0]
        self.dtype = pos.dtype

        # atomic_numbers = atomic_numbers.long()
        # num_atoms = atomic_numbers.shape[0]

        # (
        #     edge_index,
        #     edge_distance,
        #     edge_distance_vec,
        #     cell_offsets,
        #     _,  # cell offset distances
        #     neighbors,
        # ) = generate_graph(
        #     pos,
        #     cell,
        #     batch,
        #     natoms,
        #     cutoff=self.cutoff,
        #     max_neighbors=self.max_neighbors,
        #     use_pbc=self.use_pbc,
        #     otf_graph=self.otf_graph,
        # )

        ###############################################################
        # Initialize data structures
        ###############################################################

        dtype = None
        if self.wigner_fp16:
            dtype = torch.float16

        with P.record_function("precompute_wigner"):
            wigner_precomp = rot_from_xyz(
                edge_distance_vec,
                self.rh_grid.from_grid_sh_tri,
                self.jdp,
                self.mmax_list[0],
                self.rh_idx,
                # self.rh_mask,
                masker=self.masker,
                keep_full_wigner=False,
                keep_full_wigner_inv=True,
                use_rotmat=True,
                device=device,
                dtype=dtype,
            )

            # wigner_precomp = wigner_precomp._replace(
            #     wigner=self.masker(wigner_precomp.wigner, dim=2)
            # )
            # if wigner_precomp.full_wigner_inv is not None:
            #     wigner_precomp = wigner_precomp._replace(
            #         full_wigner_inv=self.masker(
            #             self.masker(wigner_precomp.full_wigner_inv, dim=1), dim=2
            #         )
            #     )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = self.sphere_embedding(atomic_numbers)  # N C (lmax+1)^2

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i, block in enumerate(self.layer_blocks):
            with P.record_function(f"LayerBlock_{i}"), ActSave.context(
                f"LayerBlock_{i}"
            ):
                x = x + block(
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    wigner_precomp,
                )

        x = self.masker(x, dim=1)

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.bmm(self.sphharm_weights[None].expand(x.shape[0], -1, -1), x)
        ActSave({"x_pt": x_pt})
        # ^ Shape: N P C

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        ActSave({"node_energy": node_energy})

        # Scale energy to help balance numerical precision w.r.t. forces
        node_energy = node_energy * 0.001

        batch_size = len(natoms)
        # energy = torch.zeros(len(data.natoms), device=device)
        # energy.index_add_(0, data.batch, node_energy.view(-1))
        with torch.cuda.amp.autocast(enabled=False):
            energy = torch.zeros(batch_size, device=device, dtype=torch.float32)
            energy = energy.index_add_(0, batch, node_energy.type_as(energy).view(-1))

        ActSave({"energy": energy})
        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x_pt, self.sphere_points)
            ActSave({"forces": forces})
            outputs["forces"] = forces

        if False and self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                "{} Time: {}\tMemory: {}\t{}".format(
                    self.counter,
                    time.time() - start_time,
                    len(data.pos),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        # self.counter = self.counter + 1

        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerBlock(torch.nn.Module):
    """
    Layer block: Perform one layer (message passing and aggregation) of the GNN

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        # SO3_grid: SO3_Grid,
        rh_grid: GridType,
        act,
        masker: Masker,
        conv_rh_grid: GridType | None = None,
    ) -> None:
        super(LayerBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels = sphere_channels
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        # self.SO3_grid = SO3_grid
        self.rh_grid = rh_grid
        self.masker = masker

        if conv_rh_grid is None:
            conv_rh_grid = rh_grid
        self.conv_rh_grid = conv_rh_grid

        # Message block
        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax_list,
            self.mmax_list,
            distance_expansion,
            max_num_elements,
            # self.SO3_grid,
            rh_grid,
            self.act,
            self.masker,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

    def load_from_escn_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.fc1_sphere.load_state_dict(filter_ckpt(state_dict, "fc1_sphere."))
        self.fc2_sphere.load_state_dict(filter_ckpt(state_dict, "fc2_sphere."))
        self.fc3_sphere.load_state_dict(filter_ckpt(state_dict, "fc3_sphere."))
        self.message_block.load_from_escn_state_dict(
            filter_ckpt(state_dict, "message_block.")
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        # mappingReduced,
    ):
        # Compute messages by performing message block
        ActSave({"x": x})
        x_message = self.message_block(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            # mappingReduced,
        )
        ActSave({"x_message": x_message})

        x_message = self.masker(x_message, dim=1)

        with (
            P.record_function("pointwise_grid_conv"),
            ActSave.context("pointwise_grid_conv"),
        ):
            ActSave({"x_message": x_message})
            if OPTIMIZE:
                ActSave(
                    {
                        "to_grid_sh_tri": self.conv_rh_grid.to_grid_sh_tri,
                        "from_grid_sh_tri": self.conv_rh_grid.from_grid_sh_tri,
                    }
                )
                x_message = fused_s2_pointwise_conv_optimized(
                    x,
                    x_message,
                    self.conv_rh_grid.to_grid_sh_tri,
                    self.conv_rh_grid.from_grid_sh_tri,
                    self.fc1_sphere.weight,
                    self.fc2_sphere.weight,
                    self.fc3_sphere.weight,
                    self.masker,
                )
            else:
                ActSave(
                    {
                        "to_grid_sh_tri": self.conv_rh_grid.to_grid_sh_tri,
                        "from_grid_sh_tri": self.conv_rh_grid.from_grid_sh_tri,
                    }
                )
                x_message = fused_s2_pointwise_conv_unoptimized(
                    x,
                    x_message,
                    self.conv_rh_grid.to_grid_sh_tri,
                    self.conv_rh_grid.from_grid_sh_tri,
                    self.fc1_sphere.weight,
                    self.fc2_sphere.weight,
                    self.fc3_sphere.weight,
                    self.masker,
                )
            ActSave({"x_message_updated": x_message})

        # Return aggregated messages
        return x_message


class MessageBlock(torch.nn.Module):
    """
    Message block: Perform message passing

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        # SO3_grid: SO3_Grid,
        rh_grid: GridType,
        act,
        masker: Masker,
    ) -> None:
        super(MessageBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        # self.SO3_grid = SO3_grid
        self.rh_grid = rh_grid
        self.num_resolutions = len(lmax_list)
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.edge_channels = edge_channels
        self.masker = masker

        # Create edge scalar (invariant to rotations) features
        self.edge_block = EdgeBlock(
            self.edge_channels,
            distance_expansion,
            max_num_elements,
            self.act,
        )
        self.so2_block_source = SO2BlockPlusPlus(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2BlockPlusPlus(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )

    def load_from_escn_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.edge_block.load_from_escn_state_dict(
            filter_ckpt(state_dict, "edge_block.")
        )
        self.so2_block_source.load_from_escn_state_dict(
            filter_ckpt(state_dict, "so2_block_source.")
        )
        self.so2_block_target.load_from_escn_state_dict(
            filter_ckpt(state_dict, "so2_block_target.")
        )

    def forward(
        self,
        x: Float[torch.Tensor, "E l_sq C"],
        atomic_numbers: Int[torch.Tensor, "N"],
        edge_distance: Float[torch.Tensor, "E"],
        edge_index: Int[torch.Tensor, "2 E"],
        SO3_edge_rot: "RotType",
        # mappingReduced,
    ):
        idx_s, idx_t = edge_index

        ###############################################################
        # Compute messages
        ###############################################################
        with P.record_function("MessageBlock"), ActSave.context("MessageBlock"):
            # Compute edge scalar features (invariant to rotations)
            # Uses atomic numbers and edge distance as inputs
            x_edge = self.edge_block(
                edge_distance,
                atomic_numbers[idx_s],  # Source atom atomic number
                atomic_numbers[idx_t],  # Target atom atomic number
            )
            ActSave({"x_edge": x_edge})
            tassert(Float[torch.Tensor, "E edge_channels"], x_edge)

            with P.record_function("select_edge_features"):
                x_source = x[idx_s]
                tassert(Float[torch.Tensor, "E l_sq C"], x_source)
                ActSave({"x_source": x_source})

                x_target = x[idx_t]
                tassert(Float[torch.Tensor, "E l_sq C"], x_target)
                ActSave({"x_target": x_target})

                del x

            with P.record_function("rotate"):
                x_source = torch.bmm(SO3_edge_rot.wigner, x_source)
                tassert(Float[torch.Tensor, "E m2l C"], x_source)

                x_source = rearrange_view(
                    x_source,
                    "E (m two_sign l) C -> E m two_sign l C",
                    m=self.mmax_list[0] + 1,
                    two_sign=2,
                    l=self.mmax_list[0] + 1,
                )
                ActSave({"x_source_rot": x_source})

                x_target = torch.bmm(SO3_edge_rot.wigner, x_target)
                tassert(Float[torch.Tensor, "E m2l C"], x_target)

                x_target = rearrange_view(
                    x_target,
                    "E (m two_sign l) C -> E m two_sign l C",
                    m=self.mmax_list[0] + 1,
                    two_sign=2,
                    l=self.mmax_list[0] + 1,
                )
                ActSave({"x_target_rot": x_target})
            if False:
                with P.record_function("rotate"):
                    x = rearrange_view(
                        x, "two_st E l_sq C -> (two_st E) l_sq C", two_st=2
                    )
                    x = torch.bmm(
                        repeat(
                            SO3_edge_rot.full_wigner,
                            "E ... -> (two_st E) ...",
                            two_st=2,
                        ),
                        x,
                    )
                    # ^ Shape: 2 E l_sq C

                    x = x[:, SO3_edge_rot.rh_idx.view(-1)]
                    x = rearrange_view(
                        x,
                        "(two_st E) (m two_sign l) C -> two_st E m two_sign l C",
                        two_st=2,
                        m=self.mmax_list[0] + 1,
                        two_sign=2,
                        l=self.mmax_list[0] + 1,
                    )

            # Compute messages
            with P.record_function("so2_block"):
                with ActSave.context("so2_block_source"):
                    x_source = self.so2_block_source(x_source, x_edge)
                    tassert(Float[torch.Tensor, "E m 2 l C"], x_source)

                    ActSave({"x_source_updated": x_source})

                with ActSave.context("so2_block_target"):
                    x_target = self.so2_block_target(x_target, x_edge)
                    tassert(Float[torch.Tensor, "E m 2 l C"], x_target)

                    ActSave({"x_target_updated": x_target})

            x = x_source + x_target
            del x_source, x_target
            x = rearrange_view(x, "E m two_sign l C -> E (m two_sign l) C")
            ActSave({"x_updated": x})

            # assert x.dtype == torch.float16
            if OPTIMIZE:
                x = fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_optimized(
                    x,
                    self.rh_grid.to_grid_sh_rh,
                    SO3_edge_rot.wigner_inv_from_grid,
                )
                ActSave({"x_updated_post_nonlinearity_and_rot_inv": x})
            else:
                x = fused_s2_pointwise_nonlinearity_and_rotate_inv_combined_unoptimized(
                    x,
                    self.rh_grid.to_grid_sh_rh,
                    self.rh_grid.from_grid_sh_tri,
                    SO3_edge_rot.full_wigner_inv,
                    self.masker,
                )
                ActSave({"x_updated_post_nonlinearity_and_rot_inv_unoptimized": x})

            with P.record_function("message_scatter"):
                # Compute the sum of the incoming neighboring messages for each target node
                # x_target._reduce_edge(idx_t, len(x.embedding))
                x = scatter(
                    x, idx_t, dim=0, dim_size=atomic_numbers.shape[0], reduce="sum"
                )
                ActSave({"x_scattered": x})

            return x


# @torch.jit.script
def _fused_so2_block(
    x: torch.Tensor,
    x_edge: torch.Tensor,
    *,
    # Params
    edge_invariant_mlp_weight: torch.Tensor,
    edge_invariant_mlp_bias: torch.Tensor,
    so2_weight_dense_1_weight: torch.Tensor,  # Float[nn.Parameter, "two_imag m l*C H"],
    so2_weight_dense_2_weight: torch.Tensor,  # Float[nn.Parameter, "two_imag m H l*C"],
):
    E, m, _, l, C = x.shape
    x = x.view(E, m, 2, l * C)
    ActSave({"x_input": x})

    x_edge = F.linear(x_edge, edge_invariant_mlp_weight, edge_invariant_mlp_bias)
    # ^ Shape: E two_imag*m*H
    x_edge = F.silu(x_edge)
    # ^ Shape: E two_imag*m*H
    # x_edge = x_edge.view(E, m, 2, -1)
    x_edge = rearrange(x_edge, "E (m two_imag H) -> E two_imag m H", two_imag=2, m=m)
    # ^ Shape: E two_imag m H
    # self.weight.shape = two_imag m l*C H
    ActSave({"x_edge": x_edge})

    # E two_imag m two_sign H
    x = torch.einsum("imhl,emsl->eimsh", so2_weight_dense_1_weight, x)
    ActSave({"x_r_post_fc1_r": x[:, 0], "x_i_post_fc1_i": x[:, 1]})
    # ^ Shape: E two_imag m two_sign H
    x = x * x_edge[..., None, :]
    ActSave({"x_r_post_mul_r": x[:, 0], "x_i_post_mul_i": x[:, 1]})
    # ^ Shape: E two_imag m two_sign H
    x = torch.einsum("eimsh,imlh->eimsl", x, so2_weight_dense_2_weight)
    ActSave({"x_r_post_fc2_r": x[:, 0], "x_i_post_fc2_i": x[:, 1]})
    # ^ Shape: E two_imag m two_sign l*C

    return x


# @torch.jit.script
def _so2_block_rest(
    x: torch.Tensor,
    E: int,
    m: int,
    l: int,
    C: int,
):
    x_r, x_i = x[:, 0], x[:, 1]
    # ^ Shape: E m two_sign (l C)

    # Unpack the -m and +m parts
    x_r_pos, x_r_neg = x_r[..., 0, :], x_r[..., 1, :]
    x_i_pos, x_i_neg = x_i[..., 0, :], x_i[..., 1, :]
    # ^ Shape: E m (l C)

    # Combine the real and imaginary parts
    x_m_pos = x_r_pos - x_i_neg
    ActSave({"x_m_r": x_m_pos})

    x_m_neg = x_r_neg + x_i_pos
    ActSave({"x_m_i": x_m_neg})
    # ^ Shape: E m (l C)

    # Pack the -m and +m parts into the sign dimension
    xm_out = torch.stack((x_m_pos, x_m_neg), dim=-2)
    # ^ Shape: E m two_sign (l C)

    xm_out = xm_out.view(E, m, 2, l, C)
    ActSave({"x_m_updated": xm_out})

    return xm_out


# @torch.jit.script
def _fused_so2_block_both(
    x: torch.Tensor,
    x_edge: torch.Tensor,
    E: int,
    m: int,
    l: int,
    C: int,
    *,
    # Params
    edge_invariant_mlp_weight: torch.Tensor,
    edge_invariant_mlp_bias: torch.Tensor,
    so2_weight_dense_1_weight: torch.Tensor,  # Float[nn.Parameter, "two_st two_imag m l*C H"],
    so2_weight_dense_2_weight: torch.Tensor,  # Float[nn.Parameter, "two_st two_imag m H l*C"],
):
    x = _fused_so2_block(
        x,
        x_edge,
        edge_invariant_mlp_weight=edge_invariant_mlp_weight,
        edge_invariant_mlp_bias=edge_invariant_mlp_bias,
        so2_weight_dense_1_weight=so2_weight_dense_1_weight,
        so2_weight_dense_2_weight=so2_weight_dense_2_weight,
    )
    x = _so2_block_rest(x, E, m, l, C=C)
    return x


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(-2)
    num_output_fmaps = tensor.size(-1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.size()[:-2]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)


def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0.0, std)


class SO2BlockPlusPlusM0(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.mmax = mmax_list[0]
        self.act = act

        num_channels_m0 = (self.mmax + 1) * self.sphere_channels

        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(num_channels_m0, self.hidden_channels, bias=False)
        self.fc2_m0 = nn.Linear(self.hidden_channels, num_channels_m0, bias=False)

    def forward(
        self,
        x_0: Float[torch.Tensor, "E two_sign l C"],
        x_edge: Float[torch.Tensor, "E C"],
    ):
        # Compute edge scalar features for m=0
        x_edge_0 = self.act(self.fc1_dist0(x_edge))
        ActSave({"x_edge_m0": x_edge_0})
        # ^ Shape: E H

        x_0 = x_0[:, signidx(+1)]
        ActSave({"x_m0": x_0})
        x_0 = rearrange_view(x_0, "E l C -> E (l C)")

        x_0 = self.fc1_m0(x_0)
        # ^ Shape: E H
        x_0 = x_0 * x_edge_0
        # ^ Shape: E H
        x_0 = self.fc2_m0(x_0)
        # ^ Shape: E (l C)

        x_0 = rearrange_view(
            x_0,
            "E (l C) -> E l C",
            l=self.mmax + 1,
            C=self.sphere_channels,
        )
        ActSave({"x_m0_updated": x_0})
        x_0 = torch.stack([x_0, torch.zeros_like(x_0)], dim=1)  # Add back "-0"

        return x_0

    # def load_from_escn_state_dict(self, state_dict: dict[str, torch.Tensor]):
    #     return self.load_state_dict(
    #         {
    #             "fc1_dist0.weight": state_dict["fc1_dist0.weight"],
    #             "fc1_dist0.bias": state_dict["fc1_dist0.bias"],
    #             "fc1_m0.weight": state_dict["fc1_m0.weight"],
    #             "fc2_m0.weight": state_dict["fc2_m0.weight"],
    #         }
    #     )


class SO2BlockPlusPlus(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    so2_weight_dense_1_weight: Float[torch.Tensor, "two_imag m H l*C"]
    so2_weight_dense_2_weight: Float[torch.Tensor, "two_imag m l*C H"]

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super().__init__()

        self.mmax = mmax_list[0]
        self.act = act
        m = self.mmax  # This is no longer mmax+1 because m=0 is handled separately
        l = self.mmax + 1
        C = sphere_channels
        H = hidden_channels

        two_imag = 2
        self.edge_invariant_mlp = nn.Linear(edge_channels, two_imag * H * m)

        # TODO: Proper initialization
        so2_weight_dense_1_weight = torch.randn((two_imag, m, H, l * C))
        _ = xavier_normal_(so2_weight_dense_1_weight)
        self.so2_weight_dense_1_weight = nn.Parameter(
            so2_weight_dense_1_weight, requires_grad=True
        )

        # two_st two_imag m l*C H
        so2_weight_dense_2_weight = torch.randn((two_imag, m, l * C, H))
        _ = xavier_normal_(so2_weight_dense_2_weight)
        self.so2_weight_dense_2_weight = nn.Parameter(
            so2_weight_dense_2_weight, requires_grad=True
        )

        self.m0_block = SO2BlockPlusPlusM0(
            sphere_channels,
            hidden_channels,
            edge_channels,
            lmax_list,
            mmax_list,
            act,
        )

    def _load_edge_invariant_weights(self, mmax, so2_conv_layers):
        edge_invariant_mlp_weights_list: list[
            Float[torch.Tensor, "two_hidden_channels edge_channels"]
        ] = []
        edge_invariant_mlp_biases_list: list[
            Float[torch.Tensor, "two_hidden_channels"]
        ] = []
        for _, layer_state_dict in so2_conv_layers:
            fc1_dist_weight = layer_state_dict["fc1_dist.weight"]
            tassert(
                Float[torch.Tensor, "two_hidden_channels edge_channels"],
                fc1_dist_weight,
            )
            edge_invariant_mlp_weights_list.append(fc1_dist_weight)

            fc1_dist_bias = layer_state_dict["fc1_dist.bias"]
            tassert(Float[torch.Tensor, "two_hidden_channels"], fc1_dist_bias)
            edge_invariant_mlp_biases_list.append(fc1_dist_bias)

        # Add zeros for ms that are not present in the state dict
        for m in range(mmax + 1, self.so2_weight_dense_1_weight.shape[1] + 1):
            edge_invariant_mlp_weights_list.append(
                torch.zeros_like(edge_invariant_mlp_weights_list[-1])
            )
            edge_invariant_mlp_biases_list.append(
                torch.zeros_like(edge_invariant_mlp_biases_list[-1])
            )

        edge_invariant_mlp_weight = rearrange(
            torch.stack(edge_invariant_mlp_weights_list),
            "m two_hidden_channels edge_channels -> (m two_hidden_channels) edge_channels",
            m=len(edge_invariant_mlp_weights_list),
        )
        edge_invariant_mlp_bias = rearrange(
            torch.stack(edge_invariant_mlp_biases_list),
            "m two_hidden_channels -> (m two_hidden_channels)",
            m=len(edge_invariant_mlp_biases_list),
        )

        return {
            "edge_invariant_mlp.weight": edge_invariant_mlp_weight,
            "edge_invariant_mlp.bias": edge_invariant_mlp_bias,
        }

    def load_from_escn_state_dict(self, state_dict: dict[str, torch.Tensor]):
        model_state_dict: dict[str, torch.Tensor] = {}

        # First, handle m=0
        model_state_dict.update(
            {
                "m0_block.fc1_dist0.weight": state_dict["fc1_dist0.weight"],
                "m0_block.fc1_dist0.bias": state_dict["fc1_dist0.bias"],
                "m0_block.fc1_m0.weight": state_dict["fc1_m0.weight"],
                "m0_block.fc2_m0.weight": state_dict["fc2_m0.weight"],
            }
        )

        # Handle m>0
        # Find the source model's mmax
        mmax_set = set[int]()
        for key in state_dict.keys():
            if not key.startswith("so2_conv."):
                continue

            mmax = int(key.split(".")[1])
            mmax_set.add(mmax)

        assert len(mmax_set) > 0, "No SO2 conv layers found in the state dict"
        mmax = max(mmax_set) + 1  # m=0 is handled separately

        # Load the weights
        so2_conv_layers = [
            ((i, m), filter_ckpt(state_dict, f"so2_conv.{i}."))
            for i, m in enumerate(range(1, mmax + 1))
        ]

        # edge_invariant_mlp
        model_state_dict.update(
            self._load_edge_invariant_weights(mmax, so2_conv_layers)
        )

        expected_lC = self.so2_weight_dense_1_weight.shape[-1]
        expected_l = self.mmax + 1
        hidden_channels = expected_lC // expected_l

        def pad(x: torch.Tensor, dim: int):
            nonlocal expected_lC, expected_l, hidden_channels
            shape = [x.shape[0], x.shape[1]]
            shape[dim] = expected_lC - shape[dim]
            return torch.cat([x, x.new_zeros(*shape)], dim=dim)

        def pad_m(x: torch.Tensor):
            expected_m = self.so2_weight_dense_1_weight.shape[1]
            return torch.cat(
                [x, x.new_zeros(x.shape[0], expected_m - x.shape[1], *x.shape[2:])],
                dim=1,
            )

        # so2_weight_dense_1_weight
        so2_weight_dense_1_weight_imag_list: list[
            Float[torch.Tensor, "hidden_channels l*sphere_channels"]
        ] = []
        so2_weight_dense_1_weight_real_list: list[
            Float[torch.Tensor, "hidden_channels l*sphere_channels"]
        ] = []
        for (i, m), layer_state_dict in so2_conv_layers:
            so2_weight_dense_1_weight_real = pad(
                layer_state_dict["fc1_r.weight"], dim=1
            )
            tassert(
                Float[torch.Tensor, "hidden_channels l_C"],
                so2_weight_dense_1_weight_real,
            )
            so2_weight_dense_1_weight_real_list.append(so2_weight_dense_1_weight_real)

            so2_weight_dense_1_weight_imag = pad(
                layer_state_dict["fc1_i.weight"], dim=1
            )
            tassert(
                Float[torch.Tensor, "hidden_channels l_C"],
                so2_weight_dense_1_weight_imag,
            )
            so2_weight_dense_1_weight_imag_list.append(so2_weight_dense_1_weight_imag)

        so2_weight_dense_1_weight_real = torch.stack(
            so2_weight_dense_1_weight_real_list
        )
        tassert(
            Float[torch.Tensor, "m hidden_channels l_C"], so2_weight_dense_1_weight_real
        )

        so2_weight_dense_1_weight_imag = torch.stack(
            so2_weight_dense_1_weight_imag_list
        )
        tassert(
            Float[torch.Tensor, "m hidden_channels l_C"], so2_weight_dense_1_weight_imag
        )

        so2_weight_dense_1_weight = torch.stack(
            (so2_weight_dense_1_weight_real, so2_weight_dense_1_weight_imag)
        )
        tassert(
            Float[torch.Tensor, "two_imag m hidden_channels l_C"],
            so2_weight_dense_1_weight,
        )
        model_state_dict["so2_weight_dense_1_weight"] = pad_m(so2_weight_dense_1_weight)

        # so2_weight_dense_2_weight
        so2_weight_dense_2_weight_imag_list: list[
            Float[torch.Tensor, "hidden_channels l*sphere_channels"]
        ] = []
        so2_weight_dense_2_weight_real_list: list[
            Float[torch.Tensor, "hidden_channels l*sphere_channels"]
        ] = []
        for _, layer_state_dict in so2_conv_layers:
            so2_weight_dense_2_weight_real = pad(
                layer_state_dict["fc2_r.weight"], dim=0
            )
            tassert(
                Float[torch.Tensor, "l_C hidden_channels"],
                so2_weight_dense_2_weight_real,
            )
            so2_weight_dense_2_weight_real_list.append(so2_weight_dense_2_weight_real)

            so2_weight_dense_2_weight_imag = pad(
                layer_state_dict["fc2_i.weight"], dim=0
            )
            tassert(
                Float[torch.Tensor, "l_C hidden_channels"],
                so2_weight_dense_2_weight_imag,
            )
            so2_weight_dense_2_weight_imag_list.append(so2_weight_dense_2_weight_imag)

        so2_weight_dense_2_weight_real = torch.stack(
            so2_weight_dense_2_weight_real_list
        )
        tassert(
            Float[torch.Tensor, "m l_C hidden_channels"], so2_weight_dense_2_weight_real
        )

        so2_weight_dense_2_weight_imag = torch.stack(
            so2_weight_dense_2_weight_imag_list
        )
        tassert(
            Float[torch.Tensor, "m l_C hidden_channels"], so2_weight_dense_2_weight_imag
        )

        so2_weight_dense_2_weight = torch.stack(
            (so2_weight_dense_2_weight_real, so2_weight_dense_2_weight_imag)
        )
        tassert(
            Float[torch.Tensor, "two_imag m l_C hidden_channels"],
            so2_weight_dense_2_weight,
        )
        model_state_dict["so2_weight_dense_2_weight"] = pad_m(so2_weight_dense_2_weight)

        self.load_state_dict(model_state_dict)

    def forward(
        self,
        x: Float[torch.Tensor, "E m two_sign l C"],
        x_edge: Float[torch.Tensor, "E C"],
    ):
        with P.record_function("SO2BlockPlusPlus"), ActSave.context("SO2BlockPlusPlus"):
            x_m0, x_mgt0 = x[:, 0], x[:, 1:]
            ActSave({"x_m0": x_m0, "x_mgt0": x_mgt0})

            # m=0
            x_m0 = self.m0_block(x_m0, x_edge)
            ActSave({"x_m0_updated": x_m0})

            # m>0
            E, m, _, l, C = x_mgt0.shape
            x_mgt0 = _fused_so2_block_both(
                x_mgt0,
                x_edge,
                E,
                m,
                l,
                C,
                edge_invariant_mlp_weight=self.edge_invariant_mlp.weight,
                edge_invariant_mlp_bias=self.edge_invariant_mlp.bias,
                so2_weight_dense_1_weight=self.so2_weight_dense_1_weight,
                so2_weight_dense_2_weight=self.so2_weight_dense_2_weight,
            )
            ActSave({"x_mgt0_updated": x_mgt0})

            x = torch.cat((x_m0[:, None], x_mgt0), dim=1).contiguous()
            return x


class EdgeBlock(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        edge_channels,
        distance_expansion,
        max_num_elements,
        act,
    ) -> None:
        super(EdgeBlock, self).__init__()
        self.in_channels = distance_expansion.num_output
        self.distance_expansion = distance_expansion
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def load_from_escn_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=True)

    def forward(self, edge_distance, source_element, target_element):
        with P.record_function("EdgeBlock"), ActSave.context("edge_block"):
            # Compute distance embedding
            x_dist = self.distance_expansion(edge_distance)
            ActSave({"x_dist_expansion": x_dist})
            x_dist = self.fc1_dist(x_dist)
            ActSave({"x_dist": x_dist})

            # Compute atomic number embeddings
            source_embedding = self.source_embedding(source_element)
            ActSave({"source_embedding": source_embedding})

            target_embedding = self.target_embedding(target_element)
            ActSave({"target_embedding": target_embedding})

            # Compute invariant edge embedding
            x_edge = self.act(source_embedding + target_embedding + x_dist)
            ActSave({"x_edge_sum": x_edge})
            x_edge = self.act(self.fc1_edge_attr(x_edge))
            ActSave({"x_edge": x_edge})

            return x_edge


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, num_channels: int, act) -> None:
        super(EnergyBlock, self).__init__()
        self.num_channels = num_channels
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def load_from_escn_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x_pt) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        node_energy = torch.mean(x_pt, dim=1)
        return node_energy


class ForceBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, num_channels: int, act) -> None:
        super(ForceBlock, self).__init__()
        self.num_channels = num_channels
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def load_from_escn_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x_pt, sphere_points) -> torch.Tensor:
        num_sphere_samples = x_pt.shape[1]
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        # x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, num_sphere_samples, 3)
        forces = torch.mean(forces, dim=1)

        return forces
