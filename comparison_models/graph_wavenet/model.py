"""
GraphWaveNet Baseline Model

Comparison model against TransGTR.

Input
-----
(B,T,N,F)

Output
------
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





class GraphWaveNetBaseline(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 64,
    ):

        super().__init__()



        self.history_length = (
            HISTORY_LENGTH
        )


        self.prediction_horizon = (
            PREDICTION_HORIZON
        )


        self.num_features = (
            NUM_INPUT_FEATURES
        )


        self.num_nodes = 9



        # --------------------------------------------------
        # Feature Projection
        #
        # F -> hidden
        # --------------------------------------------------

        self.input_projection = nn.Linear(

            self.num_features,

            hidden_dim,

        )



        # --------------------------------------------------
        # Temporal Convolution
        #
        # captures time dependency
        # --------------------------------------------------

        self.temporal_conv = nn.Conv2d(

            in_channels=hidden_dim,

            out_channels=hidden_dim,

            kernel_size=(3,1),

            padding=(1,0),

        )



        # --------------------------------------------------
        # Graph Mixing Layer
        #
        # simplified adaptive graph operation
        # --------------------------------------------------

        self.graph_layer = nn.Linear(

            self.num_nodes,

            self.num_nodes,

        )



        # --------------------------------------------------
        # Forecast Head
        #
        # hidden -> horizon
        # --------------------------------------------------

        self.output_layer = nn.Linear(

            hidden_dim,

            self.prediction_horizon,

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
        # Feature embedding
        #
        # (B,T,N,F)
        #
        # ->
        #
        # (B,T,N,D)
        # ----------------------------------------------

        x = self.input_projection(x)



        # ----------------------------------------------
        # Prepare for Conv2D
        #
        # (B,T,N,D)
        #
        # ->
        #
        # (B,D,T,N)
        # ----------------------------------------------

        x = x.permute(

            0,

            3,

            1,

            2,

        )



        x = self.temporal_conv(
            x
        )



        # ----------------------------------------------
        # Back
        #
        # (B,D,T,N)
        #
        # ->
        #
        # (B,T,N,D)
        # ----------------------------------------------

        x = x.permute(

            0,

            2,

            3,

            1,

        )


        # Last time step

        x = x[:, -1, :, :]

        # (B,N,D)



        # ----------------------------------------------
        # Graph Mixing
        # ----------------------------------------------

        x = x.permute(

            0,

            2,

            1,

        )

        # (B,D,N)


        x = self.graph_layer(x)



        x = x.permute(

            0,

            2,

            1,

        )

        # (B,N,D)



        # ----------------------------------------------
        # Forecast
        # ----------------------------------------------

        prediction = self.output_layer(x)



        # (B,N,H)

        prediction = prediction.permute(

            0,

            2,

            1,

        )


        return prediction
    
    
    # ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    print("=" * 70)

    print(
        "GraphWaveNet Baseline Test"
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



    model = GraphWaveNetBaseline()



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



    # Backpropagation Test

    loss = prediction.mean()


    loss.backward()



    print(
        "✓ Backpropagation successful"
    )



    print("=" * 70)

    print(
        "GraphWaveNet Baseline test completed successfully"
    )

    print("=" * 70)
    
    