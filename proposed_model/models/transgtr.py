"""
models/transgtr.py

Complete TransGTR Architecture

Pipeline
--------
Input
(B, T, N, F)

        │
        ▼
TrafficEmbedding

        │
        ▼
TSFormer

        │
        ▼
DynamicGraphGenerator

        │
        ▼
GraphWaveNet

        │
        ▼
PredictionHead

        │
        ▼
Prediction
(B, H, N)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import (
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    HIDDEN_DIM,
    DROPOUT,
    HISTORY_LENGTH,
    NUM_INPUT_FEATURES,
    PREDICTION_HORIZON,
)

from models.embedding import TrafficEmbedding
from models.tsformer import TSFormer
from models.structure_generator import DynamicGraphGenerator
from models.graph_wavenet import GraphWaveNet
from models.prediction_head import PredictionHead


class TransGTR(nn.Module):
    """
    Complete TransGTR Model.

    Input
    -----
    x
        Shape:
        (B, T, N, F)

    Output
    ------
    prediction
        Shape:
        (B, H, N)
    """

    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Feature Embedding
        # --------------------------------------------------

        self.embedding = TrafficEmbedding(
            max_sequence_length=HISTORY_LENGTH,
            input_dim=NUM_INPUT_FEATURES,
            embedding_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
        )

        # --------------------------------------------------
        # Temporal Encoder
        # --------------------------------------------------

        self.temporal_encoder = TSFormer(
            embedding_dim=EMBEDDING_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )

        # --------------------------------------------------
        # Dynamic Graph Generator
        # --------------------------------------------------

        self.graph_generator = DynamicGraphGenerator(
            embedding_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
        )

        # --------------------------------------------------
        # Graph WaveNet
        # --------------------------------------------------

        self.graph_wavenet = GraphWaveNet(
            embedding_dim=EMBEDDING_DIM,
            num_blocks=3,
            dropout=DROPOUT,
        )

        # --------------------------------------------------
        # Forecast Head
        # --------------------------------------------------

        self.prediction_head = PredictionHead(
            history_length=HISTORY_LENGTH,
            embedding_dim=EMBEDDING_DIM,
            prediction_horizon=PREDICTION_HORIZON,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape:
            (B, T, N, F)

        Returns
        -------
        prediction
            Shape:
            (B, H, N)
        """

        if x.ndim != 4:
            raise ValueError(
                "TransGTR expects input shape (B,T,N,F)"
            )

        # ----------------------------------------------
        # Embedding
        # ----------------------------------------------

        x = self.embedding(x)

        # ----------------------------------------------
        # Temporal-Spatial Transformer
        # ----------------------------------------------

        x = self.temporal_encoder(x)

        # ----------------------------------------------
        # Dynamic Graph Learning
        # ----------------------------------------------

        adjacency = self.graph_generator(x)

        # ----------------------------------------------
        # Graph WaveNet
        # ----------------------------------------------

        x = self.graph_wavenet(
            x,
            adjacency,
        )

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        prediction = self.prediction_head(x)

        return prediction


# ==========================================================
# Unit Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Running TransGTR Unit Test")
    print("=" * 70)

    torch.manual_seed(42)

    batch_size = 8
    history = HISTORY_LENGTH
    num_nodes = 9
    num_features = NUM_INPUT_FEATURES

    x = torch.randn(
        batch_size,
        history,
        num_nodes,
        num_features,
        requires_grad=True,
    )

    model = TransGTR()

    print(f"Model Type        : {model.__class__.__name__}")

    # --------------------------------------------------
    # Parameter Count
    # --------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters  : {total_params:,}")
    print(f"Trainable Params  : {trainable_params:,}")

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    prediction = model(x)

    print(f"Input Shape       : {tuple(x.shape)}")
    print(f"Output Shape      : {tuple(prediction.shape)}")

    expected_shape = (
        batch_size,
        PREDICTION_HORIZON,
        num_nodes,
    )

    assert prediction.shape == expected_shape, (
        f"Shape mismatch!\n"
        f"Expected: {expected_shape}\n"
        f"Received: {prediction.shape}"
    )

    print("✓ Output shape verification passed.")

    # --------------------------------------------------
    # NaN / Inf Check
    # --------------------------------------------------

    assert not torch.isnan(prediction).any(), \
        "NaN values detected in model output."

    assert not torch.isinf(prediction).any(), \
        "Inf values detected in model output."

    print("✓ Numerical stability verification passed.")

    # --------------------------------------------------
    # Backward Pass
    # --------------------------------------------------

    loss = prediction.mean()

    loss.backward()

    print("✓ Backpropagation successful.")

    # --------------------------------------------------
    # Gradient Check
    # --------------------------------------------------

    no_grad = []

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            if parameter.grad is None:

                no_grad.append(name)

    assert len(no_grad) == 0, (
        "Parameters without gradients:\n"
        + "\n".join(no_grad)
    )

    print("✓ Gradient flow verification passed.")

    # --------------------------------------------------
    # Train / Eval Mode
    # --------------------------------------------------

    model.train()

    train_prediction = model(x)

    model.eval()

    with torch.no_grad():

        eval_prediction = model(x)

    assert train_prediction.shape == eval_prediction.shape

    print("✓ Train/Eval mode verification passed.")

    # --------------------------------------------------
    # Save / Load Check
    # --------------------------------------------------

    state = model.state_dict()

    cloned_model = TransGTR()

    cloned_model.load_state_dict(state)

    cloned_model.eval()

    with torch.no_grad():

        cloned_prediction = cloned_model(x)

    assert cloned_prediction.shape == prediction.shape

    print("✓ StateDict save/load verification passed.")

    # --------------------------------------------------
    # Output Statistics
    # --------------------------------------------------

    print(f"Prediction Mean   : {prediction.mean().item():.6f}")
    print(f"Prediction Std    : {prediction.std().item():.6f}")
    print(f"Prediction Min    : {prediction.min().item():.6f}")
    print(f"Prediction Max    : {prediction.max().item():.6f}")

    print("=" * 70)
    print("✓ ALL TRANSGTR TESTS PASSED SUCCESSFULLY")
    print("=" * 70)