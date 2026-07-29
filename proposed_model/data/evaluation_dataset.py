"""
Evaluation Dataset

Testing Dataset Pipeline

Uses:
    location_1 + location_2

Same pipeline as training.

Supports:
    Horizon:
        3
        5
        10
"""


from __future__ import annotations


import torch

from torch.utils.data import Dataset


from proposed_model.data.loader import TrafficDataLoader

from proposed_model.data.window import WindowGenerator





class EvaluationDataset(Dataset):


    def __init__(

            self,

            horizon=5,

            history=12

    ):


        super().__init__()



        loader = TrafficDataLoader()



        dataframe = loader.load()



        dataframe = dataframe.fillna(0)




        generator = WindowGenerator(

            dataframe,

            history=history,

            horizon=horizon

        )



        X,Y = generator.generate()



        self.X = torch.tensor(

            X,

            dtype=torch.float32

        )


        self.Y = torch.tensor(

            Y,

            dtype=torch.float32

        )





    def __len__(self):

        return len(self.X)





    def __getitem__(

            self,

            index

    ):


        return (

            self.X[index],

            self.Y[index]

        )






# ======================================================
# TEST
# ======================================================


if __name__ == "__main__":


    print("="*70)

    print(

        "Evaluation Dataset Test"

    )

    print("="*70)



    for h in [3,5,10]:


        dataset = EvaluationDataset(

            horizon=h

        )


        x,y = dataset[0]


        print()

        print(

            "Horizon:",

            h

        )


        print(

            "Samples:",

            len(dataset)

        )


        print(

            "Input Shape:",

            x.shape

        )


        print(

            "Target Shape:",

            y.shape

        )


    print("="*70)