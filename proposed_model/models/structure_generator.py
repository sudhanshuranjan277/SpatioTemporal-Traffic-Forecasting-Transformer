"""
models/structure_generator.py

Dynamic Structure Generator

Generates a dynamic adjacency matrix from node
representations learned by TSFormer.

Input
-----
(B, T, N, D)

Output
------
(N, N)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import EMBEDDING_DIM


class DynamicGraphGenerator(nn.Module):
    """
    Dynamic Structure Generator.

    Learns the graph topology directly from
    node representations instead of using
    a fixed adjacency matrix.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization.
        """

        nn.init.xavier_uniform_(
            self.query_projection.weight
        )
        nn.init.zeros_(
            self.query_projection.bias
        )

        nn.init.xavier_uniform_(
            self.key_projection.weight
        )
        nn.init.zeros_(
            self.key_projection.bias
        )
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
    

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (B, T, N, D)"
            )

        batch_size, seq_len, num_nodes, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {embedding_dim}."
            )

        # --------------------------------------------------
        # Aggregate temporal information
        # (B, T, N, D) -> (B, N, D)
        # --------------------------------------------------
        node_features = x.mean(dim=1)

        # --------------------------------------------------
        # Linear projections
        # --------------------------------------------------
        query = self.query_projection(node_features)
        key = self.key_projection(node_features)

        # --------------------------------------------------
        # Attention score
        # (B, N, D) × (B, D, N)
        # ->
        # (B, N, N)
        # --------------------------------------------------
        scores = torch.matmul(
            query,
            key.transpose(-1, -2),
        )

        scores = scores / math.sqrt(self.embedding_dim)

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        adjacency = F.softmax(
            scores,
            dim=-1,
        )

        adjacency = self.dropout(adjacency)

        return adjacency


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

    generator = DynamicGraphGenerator()

    adjacency = generator(x)

    print("=" * 60)
    print("Input Shape      :", x.shape)
    print("Adjacency Shape  :", adjacency.shape)
    print("=" * 60)