"""
models/graph_wavenet.py

Graph WaveNet Backbone

Input
-----
Node Features
(B, T, N, D)

Adjacency
(B, N, N)

Output
------
(B, T, N, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from proposed_model.configs.config import EMBEDDING_DIM

# =============================================================================
# Graph Convolution
# =============================================================================

class GraphConv(nn.Module):
    """
    Dynamic Graph Convolution.

    Input
    -----
    x:
        (B, T, N, D)

    adjacency:
        (B, N, N)

    Output
    ------
    (B, T, N, D)
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "GraphConv expects input shape (B,T,N,D)"
            )

        if adjacency.ndim != 3:
            raise ValueError(
                "Adjacency must have shape (B,N,N)"
            )

        batch_size, history, num_nodes, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {embedding_dim}"
            )

        if adjacency.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between x and adjacency."
            )

        if adjacency.shape[1] != num_nodes or adjacency.shape[2] != num_nodes:
            raise ValueError(
                "Adjacency matrix size does not match number of nodes."
            )

        # ----------------------------------------------------
        # Dynamic graph propagation
        #
        # adjacency : (B,N,N)
        # x         : (B,T,N,D)
        #
        # output    : (B,T,N,D)
        # ----------------------------------------------------

        x = torch.einsum(
            "bij,btjd->btid",
            adjacency,
            x,
        )

        x = self.linear(x)

        return x


# =============================================================================
# Temporal Convolution
# =============================================================================

class TemporalConv(nn.Module):
    """
    Temporal Convolution Layer.

    Performs convolution only along
    the temporal dimension.
    """

    def __init__(
        self,
        embedding_dim: int,
        kernel_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size - 1, 0),
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.conv.weight)

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "TemporalConv expects input shape (B,T,N,D)"
            )

        residual_length = x.size(1)

        # ------------------------------------
        # (B,T,N,D)
        #
        # ->
        #
        # (B,D,T,N)
        # ------------------------------------

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        x = self.conv(x)

        x = self.activation(x)

        x = self.dropout(x)

        # ------------------------------------
        # Remove extra timestep introduced
        # by convolution padding
        # ------------------------------------

        current_length = x.size(2)

        if current_length > residual_length:
            x = x[:, :, :residual_length, :]

        # ------------------------------------
        # (B,D,T,N)
        #
        # ->
        #
        # (B,T,N,D)
        # ------------------------------------

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        return x
    

# =============================================================================
# Graph WaveNet Block
# =============================================================================

class GraphWaveNetBlock(nn.Module):
    """
    Single Graph WaveNet Block.

    Architecture
    ------------
    Input
        ↓
    Temporal Convolution
        ↓
    Graph Convolution
        ↓
    Dropout
        ↓
    Residual Connection
        ↓
    Layer Normalization
        ↓
    Output
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.temporal = TemporalConv(
            embedding_dim=embedding_dim,
            kernel_size=2,
            dropout=dropout,
        )

        self.graph = GraphConv(
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = nn.LayerNorm(
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape:
            (B, T, N, D)

        adjacency
            Shape:
            (B, N, N)

        Returns
        -------
        Tensor
            Shape:
            (B, T, N, D)
        """

        if x.ndim != 4:
            raise ValueError(
                "GraphWaveNetBlock expects input shape (B,T,N,D)"
            )

        if adjacency.ndim != 3:
            raise ValueError(
                "Adjacency must have shape (B,N,N)"
            )

        batch_size, _, num_nodes, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {embedding_dim}"
            )

        if adjacency.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between input and adjacency."
            )

        if adjacency.shape[1] != num_nodes or adjacency.shape[2] != num_nodes:
            raise ValueError(
                "Adjacency matrix dimensions do not match number of nodes."
            )

        # -------------------------------------------------
        # Residual Connection
        # -------------------------------------------------

        residual = x

        # -------------------------------------------------
        # Temporal Modeling
        # -------------------------------------------------

        x = self.temporal(x)

        # -------------------------------------------------
        # Spatial Graph Propagation
        # -------------------------------------------------

        x = self.graph(
            x,
            adjacency,
        )

        # -------------------------------------------------
        # Regularization
        # -------------------------------------------------

        x = self.dropout(x)

        # -------------------------------------------------
        # Residual Addition
        # -------------------------------------------------

        x = x + residual

        # -------------------------------------------------
        # Layer Normalization
        # -------------------------------------------------

        x = self.norm(x)

        return x    

# =============================================================================
# Graph WaveNet
# =============================================================================

class GraphWaveNet(nn.Module):
    """
    Graph WaveNet Backbone.

    Input
    -----
    x:
        (B, T, N, D)

    adjacency:
        (B, N, N)

    Output
    ------
    (B, T, N, D)
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.blocks = nn.ModuleList(
            [
                GraphWaveNetBlock(
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape:
            (B, T, N, D)

        adjacency
            Shape:
            (B, N, N)

        Returns
        -------
        Tensor
            Shape:
            (B, T, N, D)
        """

        if x.ndim != 4:
            raise ValueError(
                "GraphWaveNet expects input shape (B,T,N,D)"
            )

        if adjacency.ndim != 3:
            raise ValueError(
                "Adjacency must have shape (B,N,N)"
            )

        batch_size, _, num_nodes, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {embedding_dim}"
            )

        if adjacency.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between input and adjacency."
            )

        if adjacency.shape[1] != num_nodes or adjacency.shape[2] != num_nodes:
            raise ValueError(
                "Adjacency matrix dimensions do not match number of nodes."
            )

        for block in self.blocks:
            x = block(
                x,
                adjacency,
            )

        return x


# =============================================================================
# Unit Test
# =============================================================================

if __name__ == "__main__":

    batch_size = 4
    history = 12
    num_nodes = 9
    embedding_dim = EMBEDDING_DIM

    print("=" * 70)
    print("Testing GraphWaveNet")
    print("=" * 70)

    x = torch.randn(
        batch_size,
        history,
        num_nodes,
        embedding_dim,
    )

    adjacency = torch.rand(
        batch_size,
        num_nodes,
        num_nodes,
    )

    # Normalize adjacency row-wise
    adjacency = torch.softmax(
        adjacency,
        dim=-1,
    )

    model = GraphWaveNet(
        embedding_dim=embedding_dim,
        num_blocks=3,
        dropout=0.1,
    )

    output = model(
        x,
        adjacency,
    )

    print(f"Input Shape      : {x.shape}")
    print(f"Adjacency Shape  : {adjacency.shape}")
    print(f"Output Shape     : {output.shape}")

    assert output.shape == x.shape, \
        "Output shape does not match input shape."

    print("\n✓ Shape verification passed.")

    # Check for NaNs
    if torch.isnan(output).any():
        raise RuntimeError(
            "NaN values detected in GraphWaveNet output."
        )

    print("✓ NaN check passed.")

    # Backpropagation test
    output.mean().backward()

    print("✓ Backpropagation successful.")

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrainable Parameters: {total_params:,}")

    print("=" * 70)
    print("GraphWaveNet test completed successfully.")
    print("=" * 70)    