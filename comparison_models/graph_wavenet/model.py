"""
Graph WaveNet Model

Input:
(B, T, N, F)

B = Batch
T = History length
N = Nodes
F = Features


Output:

(B, H, N)

H = Forecast Horizon
"""


from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


from proposed_model.configs.config import (
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    NUM_INPUT_FEATURES
)





# ==========================================================
# Adaptive Graph Learning
# ==========================================================


class AdaptiveGraphLearner(nn.Module):


    def __init__(
        self,
        num_nodes,
        embedding_dim=10
    ):

        super().__init__()



        self.node_embedding1 = nn.Parameter(

            torch.randn(
                num_nodes,
                embedding_dim
            )

        )


        self.node_embedding2 = nn.Parameter(

            torch.randn(
                embedding_dim,
                num_nodes
            )

        )




    def forward(self):


        adjacency = torch.matmul(

            self.node_embedding1,

            self.node_embedding2

        )



        adjacency = F.softmax(

            F.relu(adjacency),

            dim=1

        )


        return adjacency







# ==========================================================
# Graph Convolution
# ==========================================================


class GraphConvolution(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels
    ):


        super().__init__()



        self.linear = nn.Linear(

            in_channels,

            out_channels

        )




    def forward(
        self,
        x,
        adjacency
    ):


        """
        x:

        (B,T,N,F)

        """


        x = torch.einsum(

            "nm,btmf->btnf",

            adjacency,

            x

        )



        x = self.linear(x)



        return x







# ==========================================================
# Graph WaveNet
# ==========================================================


class GraphWaveNet(nn.Module):


    def __init__(

        self,

        num_nodes=3,

        hidden_dim=64,

        horizon=PREDICTION_HORIZON

    ):


        super().__init__()



        self.num_nodes = num_nodes

        self.horizon = horizon




        # Adaptive adjacency

        self.graph_learner = AdaptiveGraphLearner(

            num_nodes

        )




        # Temporal convolution


        self.temporal_conv = nn.Conv2d(

            in_channels=NUM_INPUT_FEATURES,

            out_channels=hidden_dim,

            kernel_size=(3,1),

            padding=(1,0)

        )




        self.graph_conv = GraphConvolution(

            hidden_dim,

            hidden_dim

        )




        self.output_layer = nn.Linear(

            hidden_dim,

            horizon

        )







    def forward(self,x):


        """

        Input:

        B,T,N,F


        Output:

        B,H,N

        """



        if x.ndim !=4:


            raise ValueError(

                "Expected input shape (B,T,N,F)"

            )




        # rearrange

        x = x.permute(

            0,

            3,

            1,

            2

        )


        # B,F,T,N


        x = self.temporal_conv(x)



        # B,Hid,T,N


        x = x.permute(

            0,

            2,

            3,

            1

        )


        # B,T,N,Hid



        adjacency = self.graph_learner()



        x = self.graph_conv(

            x,

            adjacency

        )



        # Last timestep


        x = x[:,-1,:,:]



        # B,N,Hid


        x = self.output_layer(

            x

        )



        # B,N,H


        x = x.permute(

            0,

            2,

            1

        )


        return x







# ==========================================================
# Test
# ==========================================================


if __name__ == "__main__":


    print("="*70)

    print(
        "Graph WaveNet Test"
    )

    print("="*70)



    batch = 4


    nodes = 3



    dummy = torch.randn(

        batch,

        HISTORY_LENGTH,

        nodes,

        NUM_INPUT_FEATURES

    )



    model = GraphWaveNet(

        num_nodes=nodes,

        horizon=5

    )



    output = model(dummy)



    print(

        "Input:",

        dummy.shape

    )



    print(

        "Output:",

        output.shape

    )



    assert output.shape == (

        batch,

        5,

        nodes

    )



    print(
        "✓ Graph WaveNet working"
    )