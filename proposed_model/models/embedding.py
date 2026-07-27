"""
models/embedding.py

Traffic Embedding Module

Pipeline
--------
Raw Features
      │
      ▼
Feature Projection
      │
      ▼
Positional Embedding
      │
      ▼
Dropout
      │
      ▼
Embedded Features

Input Shape
-----------
(B, T, N, F)

Output Shape
------------
(B, T, N, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import (
    EMBEDDING_DIM,
    NUM_INPUT_FEATURES,
)


class FeatureEmbedding(nn.Module):
    """
    Projects raw traffic features into embedding space.
    """

    def __init__(
        self,
        input_dim: int = NUM_INPUT_FEATURES,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            embedding_dim,
            bias=True,
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.projection.weight)

        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B,T,N,F), got {tuple(x.shape)}"
            )

        return self.projection(x)


class PositionalEmbedding(nn.Module):
    """
    Learnable temporal positional embedding.
    """

    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                max_sequence_length,
                1,
                embedding_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence_length = x.size(1)

        return x + self.position_embedding[:, :sequence_length]


class TrafficEmbedding(nn.Module):
    """
    Complete embedding block.

    Input
    -----
    (B, T, N, F)

    Output
    ------
    (B, T, N, D)
    """

    def __init__(
        self,
        max_sequence_length: int,
        input_dim: int = NUM_INPUT_FEATURES,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_embedding = FeatureEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.feature_embedding(x)

        x = self.position_embedding(x)

        x = self.dropout(x)

        return x