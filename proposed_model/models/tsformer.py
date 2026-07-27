"""
models/tsformer.py

Temporal-Spatial Transformer Encoder

Input Shape
-----------
(B, T, N, D)

Output Shape
------------
(B, T, N, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import EMBEDDING_DIM


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed Forward Network.

    Input
    -----
    (B, T, D)

    Output
    ------
    (B, T, D)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.linear2(x)

        x = self.dropout(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.ffn = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        x = self.norm1(
            residual + x
        )

        residual = x

        x = self.ffn(x)

        x = self.norm2(
            residual + x
        )

        return x


class TSFormer(nn.Module):
    """
    Temporal-Spatial Transformer.

    Expected Input
    --------------
    (B, T, N, D)

    Returns
    -------
    (B, T, N, D)
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, N, D)

        Returns
        -------
        (B, T, N, D)
        """

        if x.ndim != 4:
            raise ValueError(
                "TSFormer expects input of shape (B, T, N, D)"
            )

        batch_size, seq_len, num_nodes, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {embedding_dim}."
            )

        # (B, T, N, D) -> (B*N, T, D)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size * num_nodes,
            seq_len,
            embedding_dim,
        )

        for layer in self.layers:
            x = layer(x)

        # (B*N, T, D) -> (B, T, N, D)
        x = x.reshape(
            batch_size,
            num_nodes,
            seq_len,
            embedding_dim,
        )

        x = x.permute(0, 2, 1, 3)

        return x


if __name__ == "__main__":

    batch_size = 8
    history = 12
    nodes = 9
    embedding_dim = EMBEDDING_DIM

    x = torch.randn(
        batch_size,
        history,
        nodes,
        embedding_dim,
    )

    model = TSFormer()

    output = model(x)

    print("=" * 60)
    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)
    print("=" * 60)    
    
    
    
    
    