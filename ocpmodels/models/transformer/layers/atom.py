import torch
import torch.nn as nn


class AtomEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_atom_index=84, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(max_atom_index, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)

        nn.init.uniform_(
            self.embeddings.weight, a=-torch.sqrt(3), b=torch.sqrt(3)
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.dropout_layer(x)
        return x
