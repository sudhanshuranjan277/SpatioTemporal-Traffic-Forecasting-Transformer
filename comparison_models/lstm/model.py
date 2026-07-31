"""
LSTM Baseline Model

Dynamic Forecast Horizon Support

Input:
(B,T,N,F)

Output:
(B,H,N)
"""


from __future__ import annotations


import torch
import torch.nn as nn
import pandas as pd


from proposed_model.configs.config import (
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    NUM_INPUT_FEATURES,
    DATASET_FILES,
)




def get_num_nodes():

    dataframe = pd.read_csv(
        DATASET_FILES[0]
    )

    return dataframe["junction_id"].nunique()





class LSTMBaseline(nn.Module):


    def __init__(
        self,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        horizon=None,
    ):

        super().__init__()



        self.history_length = HISTORY_LENGTH


        self.prediction_horizon = (

            horizon
            if horizon is not None
            else PREDICTION_HORIZON

        )


        self.input_dim = NUM_INPUT_FEATURES


        self.num_nodes = get_num_nodes()



        self.lstm = nn.LSTM(

            input_size=
            (
                self.num_nodes
                *
                self.input_dim
            ),

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=
            (
                dropout
                if num_layers > 1
                else 0
            )

        )



        self.output_layer = nn.Linear(

            hidden_dim,

            self.prediction_horizon
            *
            self.num_nodes

        )





    def forward(self,x):


        if x.ndim != 4:

            raise ValueError(
                "Expected input shape (B,T,N,F)"
            )



        batch = x.size(0)



        x = x.reshape(

            batch,

            self.history_length,

            -1

        )



        output,_ = self.lstm(x)



        output = output[:,-1,:]



        output = self.output_layer(
            output
        )



        output = output.reshape(

            batch,

            self.prediction_horizon,

            self.num_nodes

        )


        return output