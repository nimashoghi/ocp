from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data

from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc
from ocpmodels.models.transformer.util.coalesce import coalesce
from torch_geometric.utils import to_dense_adj, remove_self_loops


def calc_neighbor_information(batch: Batch, cutoff=1e1000, max_radius=1000):
    edge_index, cell_offsets, neighbors = radius_graph_pbc(
        batch, cutoff, max_radius, d
    )
    batch.edge_index = edge_index
    batch.cell_offsets = cell_offsets
    batch.neighbors = neighbors

    out = get_pbc_distances(
        batch.pos,
        batch.edge_index,
        batch.cell,
        batch.cell_offsets,
        batch.neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )
    return batch, out


def get_relative_positions(
    batch: List[Data],
    cutoff=1e1000,
    max_radius=1000,
    max_num_nodes=512,
    position_in_dims=3,
):
    d = batch[0].pos.device
    batch_size = len(batch)
    num_nodes = min(max_num_nodes, max(len(x.pos) for x in batch))
    adjacency_matrix = torch.zeros(
        (batch_size, num_nodes, num_nodes, position_in_dims)
    ).to(d)
    for i, data in enumerate(batch):
        last_index = len(data.pos)
        data, out = calc_neighbor_information(data, cutoff, max_radius)
        edge_index, distances = out["edge_index"], out["distance_vec"]
        edge_index, distances = remove_self_loops(edge_index, distances)
        dense_adj = to_dense_adj(edge_index, edge_attr=distances).to(d)
        adjacency_matrix[i, :last_index, :last_index] = dense_adj
    return adjacency_matrix

    # out, batch = calc_neighbor_information(batch, cutoff, max_radius)
    # edge_index = out["edge_index"]
    # distances = out["distance_vec"]
    # edge_index, distances = remove_self_loops(edge_index, distances)
    # edge_index, distances = coalesce(
    #     edge_index,
    #     distances,
    #     num_nodes=num_nodes,
    #     reduce="min",
    # )
    # adjacency_matrix = to_dense_adj(edge_index, edge_attr=distances)
    # return adjacency_matrix


def convert_graph_to_tensor(batch: List[Data]):
    d = batch[0].pos.device
    atomic_numbers = pad_sequence([x.atomic_numbers.long() for x in batch]).to(
        d
    )
    positions = pad_sequence([x.pos for x in batch]).to(d)
    unit_cell_info = pad_sequence(
        [x.cell.flatten().repeat((len(x.pos), 1)) for x in batch]
    ).to(d)
    key_padding_mask = (atomic_numbers != 0).to(d)
    return atomic_numbers, positions, unit_cell_info, key_padding_mask
