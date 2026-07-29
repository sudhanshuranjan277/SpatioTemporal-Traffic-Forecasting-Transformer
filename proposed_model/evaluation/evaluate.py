"""
Evaluation Dataset

Testing Dataset Pipeline

Uses:
    location_1 + location_2

Total Nodes:
    9


Supports Forecast Horizon:

    3
    5
    10


Output:

X:
(history, nodes, features)

Y:
(horizon, nodes)

"""


from __future__ import annotations


import numpy as np
import torch


from torch.utils.data import Dataset



from proposed_model.data.loader import TrafficDataLoader


from proposed_model.configs.config import (

    FEATURE_COLUMNS,

    TARGET_COLUMN

)






class EvaluationDataset(Dataset):


    def __init__(

            self,

            horizon=5,

            history=12

    ):


        super().__init__()



        self.horizon = horizon

        self.history = history





        # ==================================================
        # Load Combined Dataset
        # ==================================================


        loader = TrafficDataLoader()



        dataframe = loader.load()



        dataframe = dataframe.fillna(0)






        # ==================================================
        # Create Windows
        # ==================================================


        X,Y = self.create_windows(

            dataframe

        )



        self.X = torch.tensor(

            X,

            dtype=torch.float32

        )


        self.Y = torch.tensor(

            Y,

            dtype=torch.float32

        )







    # ======================================================
    # Window Generator
    # ======================================================


    def create_windows(

            self,

            dataframe

    ):



        X = []

        Y = []



        nodes = sorted(

            dataframe["junction_id"].unique()

        )



        feature_data = []

        target_data = []





        for node in nodes:



            node_df = dataframe[

                dataframe["junction_id"] == node

            ]



            node_df = node_df.sort_values(

                "simulation_time"

            )



            feature_data.append(

                node_df[FEATURE_COLUMNS].values

            )



            target_data.append(

                node_df[TARGET_COLUMN].values

            )






        feature_data = np.array(

            feature_data

        )



        target_data = np.array(

            target_data

        )



        total_steps = feature_data.shape[1]






        for i in range(

            total_steps

            -

            self.history

            -

            self.horizon

        ):



            # -------------------------
            # Input
            # -------------------------


            x = feature_data[

                :,

                i:i+self.history,

                :

            ]



            # nodes,history,features

            x = np.transpose(

                x,

                (1,0,2)

            )





            # -------------------------
            # Target
            # -------------------------


            y = target_data[

                :,

                i+self.history:

                i+self.history+self.horizon

            ]



            # nodes,horizon

            y = np.transpose(

                y,

                (1,0)

            )



            X.append(

                x

            )


            Y.append(

                y

            )




        return (

            np.array(X),

            np.array(Y)

        )







    # ======================================================
    # Dataset API
    # ======================================================


    def __len__(self):


        return len(

            self.X

        )





    def __getitem__(

            self,

            index

    ):


        return (

            self.X[index],

            self.Y[index]

        )







# ======================================================
# Test
# ======================================================


if __name__ == "__main__":


    print("="*70)

    print(

        "Combined Evaluation Dataset Test"

    )

    print("="*70)




    for h in [3,5,10]:


        dataset = EvaluationDataset(

            horizon=h

        )


        x,y = dataset[0]



        print()

        print(

            f"Horizon: {h}"

        )


        print(

            "Samples:",

            len(dataset)

        )


        print(

            "Input:",

            x.shape

        )


        print(

            "Target:",

            y.shape

        )


    print("="*70)