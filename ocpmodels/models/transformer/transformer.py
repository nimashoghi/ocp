import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocpmodels.models.transformer.layers.transformer import (
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from torch.nn.utils.rnn import pad_sequence

from ocpmodels.common.registry import registry
from ocpmodels.models.transformer.layers.atom import AtomEmbedding
from ocpmodels.models.transformer.util.data import convert_graph_to_tensor


@registry.register_model("transformer")
class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        atomic_number_embedding_dim = self.config.get(
            "atomic_number_embedding_dim", 128
        )
        position_embedding_dim = self.config.get("position_embedding_dim", 64)
        node_embedding_dim = self.config.get("node_embedding_dim", 512)

        self.position_mlp = nn.Linear(
            in_features=self.config.get("position_in_dims", 3),
            out_features=position_embedding_dim,
        )
        self.embedding = AtomEmbedding(
            atomic_number_embedding_dim,
            max_atom_index=self.config.get("max_atom_index", 84),
            dropout=self.config.get("embedding_dropout", 0.2),
        )
        self.embedding_bilinear = nn.Bilinear(
            in1_features=atomic_number_embedding_dim,
            in2_features=position_embedding_dim,
            out_features=node_embedding_dim,
        )

        self_attn_kwargs = dict(
            d_model=node_embedding_dim,
            nhead=self.config.get("nhead", 8),
            dropout=self.config.get("dropout", 0.1),
        )
        encoder_layer = TransformerEncoderLayer(**self_attn_kwargs)
        if self.config.get("use_relative_positions", False):
            self._add_relative_positions(encoder_layer, **self_attn_kwargs)
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers=self.config.get("num_layers", 6)
        )
        self.output = nn.Linear(
            in_features=node_embedding_dim,
            out_features=self.config.get("force_pred_dims", 3),
        )

        self._init_weights()

    def _add_relative_positions(
        self, encoder_layer: nn.TransformerEncoderLayer, **kwargs
    ):
        encoder_layer.self_attn = MultiheadAttention(**kwargs)

    def _init_weights(self):
        initrange = 0.1
        self.embedding.embedding.weight.data.uniform_(-initrange, initrange)

        self.position_mlp.bias.data.zero_()
        self.position_mlp.weight.data.uniform_(-initrange, initrange)

        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def _convert_data_to_tensor(self, batch):
        atomic_numbers, positions, mask = convert_graph_to_tensor(batch, device=self.device)

        return (
            self.embedding_bilinear(
                self.embedding(atomic_numbers), self.position_mlp(positions)
            ),
            mask,
        )

    def forward(self, batch):
        x, mask = self._convert_data_to_tensor(batch)
        x = self.encoder(
            x, src_key_padding_mask=(~mask).transpose(1, 0)
        ) * math.sqrt(
            self.d_model
        )  # "mask" is true if an element is not padding, so we need to negate it to get the mask for padding
        x = F.relu(x)
        x = self.output(x)

        return x, mask.unsqueeze(-1)
