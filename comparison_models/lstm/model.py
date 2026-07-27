"""
LSTM Baseline Model

Used for comparison against TransGTR.

Input:
(B, T, N, F)

B = Batch Size
T = History Length
N = Number of Nodes
F = Number of Features

Output:
(B, H, N)

H = Prediction Horizon
"""


from __future__ import annotations


import torch
import torch.nn as nn


from proposed_model.configs.config import (
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    NUM_INPUT_FEATURES,
)





class LSTMBaseline(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):

        super().__init__()



        self.history_length = HISTORY_LENGTH

        self.prediction_horizon = (
            PREDICTION_HORIZON
        )



        self.input_dim = (
            NUM_INPUT_FEATURES
        )



        self.hidden_dim = hidden_dim



        # --------------------------------------------------
        # LSTM Encoder
        #
        # Input:
        # (B,T,N,F)
        #
        # Convert:
        # (B,T,N*F)
        # --------------------------------------------------


        self.lstm = nn.LSTM(

            input_size=(
                self.input_dim
                *
                9
            ),

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout
            if num_layers > 1
            else 0,

        )



        # --------------------------------------------------
        # Forecast Head
        #
        # hidden_dim
        #       |
        #       ↓
        # H*N
        #
        # reshape:
        # (B,H,N)
        # --------------------------------------------------


        self.output_layer = nn.Linear(

    hidden_dim,

    self.prediction_horizon * 9,

)



    def forward(
        self,
        x: torch.Tensor,
    ):


        if x.ndim != 4:

            raise ValueError(
                "Expected input shape (B,T,N,F)"
            )


        batch_size = x.size(0)



        # ----------------------------------------------
        # Flatten nodes and features
        # ----------------------------------------------

        x = x.reshape(

            batch_size,

            self.history_length,

            -1,

        )



        # ----------------------------------------------
        # LSTM
        # ----------------------------------------------

        output, _ = self.lstm(x)



        # Take last time step

        output = output[:, -1, :]



        # ----------------------------------------------
        # Forecast
        # ----------------------------------------------

        output = self.output_layer(
            output
        )



        output = output.reshape(

            batch_size,

            self.prediction_horizon,

            9,

        )



        return output
    # ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    print("=" * 70)

    print(
        "LSTM Baseline Test"
    )

    print("=" * 70)



    batch_size = 4

    nodes = 9



    dummy_input = torch.randn(

        batch_size,

        HISTORY_LENGTH,

        nodes,

        NUM_INPUT_FEATURES,

    )



    model = LSTMBaseline()



    output = model(
        dummy_input
    )



    print(
        "Input Shape :",
        dummy_input.shape
    )


    print(
        "Output Shape:",
        output.shape
    )



    expected_shape = (

        batch_size,

        PREDICTION_HORIZON,

        nodes,

    )



    assert output.shape == expected_shape, (

        f"Expected {expected_shape}, "
        f"got {output.shape}"

    )



    print(
        "✓ Output shape verification passed"
    )



    parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )



    print(
        f"Trainable Parameters: {parameters:,}"
    )



    print("=" * 70)

    print(
        "LSTM Baseline test completed"
    )

    print("=" * 70)
    