"""
Traffic Forecasting Transformer Model

Input:

(batch, seq_len, nodes, features)


Output:

(batch, horizon, nodes)

"""


import torch
import torch.nn as nn





class TrafficTransformer(nn.Module):


    def __init__(
        self,
        input_features=13,
        d_model=64,
        n_heads=4,
        num_layers=2,
        horizon=5,
        nodes=3
    ):


        super().__init__()



        self.nodes = nodes

        self.horizon = horizon



        self.input_projection = nn.Linear(

            input_features,

            d_model

        )




        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=n_heads,

            batch_first=True

        )



        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )




        self.output_layer = nn.Linear(

            d_model,

            horizon

        )







    def forward(self,x):


        # x:

        # batch, seq, nodes, features



        batch,seq,nodes,features = x.shape



        x = x.reshape(

            batch*nodes,

            seq,

            features

        )



        x = self.input_projection(x)



        x = self.transformer(x)



        # last time step

        x = x[:,-1,:]



        x = self.output_layer(x)



        x = x.reshape(

            batch,

            nodes,

            self.horizon

        )



        x = x.permute(

            0,

            2,

            1

        )


        return x







if __name__=="__main__":


    model = TrafficTransformer()



    sample=torch.randn(

        4,

        12,

        3,

        13

    )


    output=model(sample)


    print(

        "Input:",

        sample.shape

    )


    print(

        "Output:",

        output.shape

    )