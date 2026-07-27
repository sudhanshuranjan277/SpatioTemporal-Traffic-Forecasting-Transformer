"""
GRU Baseline Model

Used for comparison with TransGTR.

Input:
(B,T,N,F)

Output:
(B,H,N)
"""


from __future__ import annotations


import torch
import torch.nn as nn


from proposed_model.configs.config import (
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    NUM_INPUT_FEATURES,
)





class GRUBaseline(nn.Module):

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
        # GRU Encoder
        #
        # Input:
        # (B,T,N,F)
        #
        # Convert:
        # (B,T,N*F)
        # --------------------------------------------------

        self.gru = nn.GRU(

            input_size=
                self.input_dim * 9,

            hidden_size=
                hidden_dim,

            num_layers=
                num_layers,

            batch_first=True,

            dropout=
                dropout
                if num_layers > 1
                else 0,

        )



        # --------------------------------------------------
        # Forecast Head
        #
        # hidden_dim -> H*N
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



        # Flatten nodes + features

        x = x.reshape(

            batch_size,

            self.history_length,

            -1,

        )



        output, _ = self.gru(x)


        # Last temporal representation

        output = output[:, -1, :]


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

    num_nodes = 9



    # Dummy Input
    #
    # (B,T,N,F)
    #

    x = torch.randn(

        batch_size,

        HISTORY_LENGTH,

        num_nodes,

        NUM_INPUT_FEATURES,

    )



    model = GRUBaseline()



    prediction = model(
        x
    )



    print(
        "Input Shape :",
        x.shape
    )


    print(
        "Output Shape:",
        prediction.shape
    )



    # Expected:
    #
    # (B,H,N)
    #

    expected_shape = (

        batch_size,

        PREDICTION_HORIZON,

        num_nodes,

    )



    assert prediction.shape == expected_shape, (

        f"Expected {expected_shape}, "
        f"got {prediction.shape}"

    )



    print(
        "✓ Output shape verification passed"
    )



    # Parameter Count

    parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )



    print(
        f"Trainable Parameters: {parameters:,}"
    )



    # Backward Test

    loss = prediction.mean()


    loss.backward()



    print(
        "✓ Backpropagation successful"
    )



    print("=" * 70)

    print(
        "LSTM Baseline test completed successfully"
    )

    print("=" * 70)
    
    
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

    num_nodes = 9



    # Dummy Input
    #
    # (B,T,N,F)
    #

    x = torch.randn(

        batch_size,

        HISTORY_LENGTH,

        num_nodes,

        NUM_INPUT_FEATURES,

    )



    model = GRUBaseline()


    prediction = model(
        x
    )



    print(
        "Input Shape :",
        x.shape
    )


    print(
        "Output Shape:",
        prediction.shape
    )



    # Expected:
    #
    # (B,H,N)
    #

    expected_shape = (

        batch_size,

        PREDICTION_HORIZON,

        num_nodes,

    )



    assert prediction.shape == expected_shape, (

        f"Expected {expected_shape}, "
        f"got {prediction.shape}"

    )



    print(
        "✓ Output shape verification passed"
    )



    # Parameter Count

    parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )



    print(
        f"Trainable Parameters: {parameters:,}"
    )



    # Backward Test

    loss = prediction.mean()


    loss.backward()



    print(
        "✓ Backpropagation successful"
    )



    print("=" * 70)

    print(
        "LSTM Baseline test completed successfully"
    )

    print("=" * 70)