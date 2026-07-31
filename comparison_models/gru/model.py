"""
GRU Baseline Model

Input:
(B,T,N,F)

Output:
(B,H,N)

B = Batch
T = History Length
N = Nodes
F = Features
H = Forecast Horizon
"""


from __future__ import annotations


import torch
import torch.nn as nn

import pandas as pd


from proposed_model.configs.config import (
    HISTORY_LENGTH,
    NUM_INPUT_FEATURES,
    DATASET_FILES,
    PREDICTION_HORIZON,
)



# ======================================================
# Dynamic Node Detection
# ======================================================


def get_num_nodes():

    dataframe = pd.read_csv(
        DATASET_FILES[0]
    )

    return dataframe["junction_id"].nunique()



# ======================================================
# GRU Model
# ======================================================


class GRUBaseline(nn.Module):


    def __init__(
        self,
        horizon=None,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()



        self.history_length = HISTORY_LENGTH


        self.input_dim = NUM_INPUT_FEATURES


        self.num_nodes = get_num_nodes()



        self.horizon = (

            horizon

            if horizon is not None

            else PREDICTION_HORIZON

        )



        self.gru = nn.GRU(

            input_size=(

                self.num_nodes *
                self.input_dim

            ),

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=(

                dropout
                if num_layers > 1
                else 0

            )

        )



        self.output_layer = nn.Linear(

            hidden_dim,

            self.horizon *
            self.num_nodes

        )




    def forward(self,x):


        if x.ndim != 4:

            raise ValueError(
                "Expected input (B,T,N,F)"
            )



        batch = x.size(0)



        x = x.reshape(

            batch,

            self.history_length,

            -1

        )



        output,_ = self.gru(x)



        output = output[:,-1,:]



        output = self.output_layer(
            output
        )



        output = output.reshape(

            batch,

            self.horizon,

            self.num_nodes

        )


        return output





# ======================================================
# Test
# ======================================================


if __name__ == "__main__":


    nodes = get_num_nodes()


    model = GRUBaseline(

        horizon=5

    )


    x = torch.randn(

        4,

        HISTORY_LENGTH,

        nodes,

        NUM_INPUT_FEATURES

    )


    y = model(x)



    print(
        "Input:",
        x.shape
    )


    print(
        "Output:",
        y.shape
    )