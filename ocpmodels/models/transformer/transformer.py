import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.models.transformer.layers.atom import AtomEmbedding
from ocpmodels.models.transformer.layers.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from ocpmodels.models.transformer.util.data import (
    convert_graph_to_tensor,
    get_relative_positions,
)


@registry.register_model("transformer")
class Transformer(BaseModel):
    def __init__(
        self, num_atoms=None, bond_feat_dim=None, num_targets=None, **kwargs
    ):
        super().__init__(
            num_atoms=num_atoms,
            bond_feat_dim=bond_feat_dim,
            num_targets=num_targets,
        )

        self.config = kwargs

        atomic_number_embedding_dim = self.config.get(
            "atomic_number_embedding_dim", 128
        )
        position_embedding_dim = self.config.get("position_embedding_dim", 64)
        unit_cell_embedding_dim = self.config.get(
            "unit_cell_embedding_dim", 32
        )
        self.node_embedding_dim = self.config.get("node_embedding_dim", 768)

        self.position_mlp = nn.Linear(
            in_features=self.config.get("position_in_dims", 3),
            out_features=position_embedding_dim,
        )
        self.unit_cell_mlp = nn.Linear(
            in_features=self.config.get("unit_cell_in_dims", 9),
            out_features=unit_cell_embedding_dim,
        )
        self.embedding = AtomEmbedding(
            atomic_number_embedding_dim,
            max_atom_index=self.config.get("max_atom_index", 84),
            dropout=self.config.get("embedding_dropout", 0.2),
        )
        self.embedding_bilinear = nn.Bilinear(
            in1_features=position_embedding_dim,
            in2_features=unit_cell_embedding_dim,
            out_features=self.config.get("embedding_bottleneck_dim", 512),
        )
        self.embedding_bilinear2 = nn.Bilinear(
            in1_features=self.config.get("embedding_bottleneck_dim", 512),
            in2_features=atomic_number_embedding_dim,
            out_features=self.node_embedding_dim,
        )

        self_attn_kwargs = dict(
            d_model=self.node_embedding_dim,
            nhead=self.config.get("nhead", 8),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "gelu"),
        )
        encoder_layer = TransformerEncoderLayer(**self_attn_kwargs)
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers=self.config.get("num_layers", 6)
        )
        self.output = nn.Linear(
            in_features=self.node_embedding_dim,
            out_features=self.config.get("force_pred_dims", 3),
        )

        self._init_weights()
        self._activation_fn = encoder_layer.activation

    def _init_weights(self):
        initrange = 0.1

        self.embedding_bilinear.weight.data.uniform_(-initrange, initrange)
        self.embedding_bilinear.bias.data.fill_(0)
        self.embedding_bilinear2.weight.data.uniform_(-initrange, initrange)
        self.embedding_bilinear2.bias.data.fill_(0)
        self.position_mlp.weight.data.uniform_(-initrange, initrange)
        self.position_mlp.bias.data.fill_(0)
        self.unit_cell_mlp.weight.data.uniform_(-initrange, initrange)
        self.unit_cell_mlp.bias.data.fill_(0)
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.fill_(0)
        self.encoder.apply(self._init_weights_transformer)

    def _init_weights_transformer(self, module):
        initrange = 0.1
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-initrange, initrange)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def _convert_data_to_tensor(self, batch: Batch):
        batch = batch.to_data_list()
        (
            atomic_numbers,
            positions,
            unit_cell_info,
            mask,
        ) = convert_graph_to_tensor(batch)
        relative_positions = (
            get_relative_positions(
                batch,
                max_num_nodes=self.config.get("max_num_nodes", 512),
                position_in_dims=self.config.get("position_in_dims", 3),
            )
            if self.config.get("use_relative_positions", False)
            else None
        )

        return (
            self._activation_fn(
                self.embedding_bilinear2(
                    self._activation_fn(
                        self.embedding_bilinear(
                            self._activation_fn(self.position_mlp(positions)),
                            self._activation_fn(
                                self.unit_cell_mlp(unit_cell_info)
                            ),
                        )
                    ),
                    self.embedding(atomic_numbers),
                )
            ),
            relative_positions,
            mask,
        )

    def forward(self, batch: Batch):
        x, relative_positions, mask = self._convert_data_to_tensor(batch)
        x = self.encoder(
            x,
            relative_positions=relative_positions,
            src_key_padding_mask=(~mask).transpose(1, 0),
        ) * math.sqrt(
            self.node_embedding_dim
        )  # "mask" is true if an element is not padding, so we need to negate it to get the mask for padding
        x = self._activation_fn(x)
        x = self.output(x)

        energy_stub = torch.full((x.shape[1], 1), 0.0, device=x.device)

        return energy_stub, x
