"""
data/dataset.py

PyTorch Dataset for Traffic Forecasting

Supports Dynamic Forecast Horizon

Example:

TrafficDataset(
    split="train",
    horizon=3
)

TrafficDataset(
    split="train",
    horizon=8
)

"""


from __future__ import annotations


from typing import Tuple


import numpy as np
import torch


from torch.utils.data import Dataset



from proposed_model.configs.config import (
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


from proposed_model.data.loader import (
    TrafficDataLoader,
)


from proposed_model.data.window import (
    WindowGenerator,
)





class TrafficDataset(Dataset):


    VALID_SPLITS = (

        "train",

        "validation",

        "test",

    )



    def __init__(

        self,

        split: str = "train",

        horizon=None,

    ):


        super().__init__()



        if split not in self.VALID_SPLITS:


            raise ValueError(

                f"Invalid split: {split}"

            )



        self.split = split


        self.horizon = horizon





        # ==================================================
        # Load Data
        # ==================================================


        loader = TrafficDataLoader()



        dataframe = loader.load()



        dataframe = dataframe.fillna(0)





        # ==================================================
        # Dynamic Sliding Window
        # ==================================================


        window_generator = WindowGenerator(

            dataframe,

            horizon=self.horizon

        )



        features, targets = (

            window_generator.generate()

        )





        # ==================================================
        # Validation
        # ==================================================


        if np.isnan(features).any():

            raise ValueError(

                "NaN found in features"

            )



        if np.isnan(targets).any():

            raise ValueError(

                "NaN found in targets"

            )



        if np.isinf(features).any():

            raise ValueError(

                "Inf found in features"

            )



        if np.isinf(targets).any():

            raise ValueError(

                "Inf found in targets"

            )





        # ==================================================
        # Tensor Conversion
        # ==================================================


        self.features = torch.tensor(

            features,

            dtype=torch.float32

        )



        self.targets = torch.tensor(

            targets,

            dtype=torch.float32

        )




        # ==================================================
        # Split
        # ==================================================


        self._create_split()






    # ======================================================
    # Dataset Split
    # ======================================================


    def _create_split(self):


        total_samples = len(

            self.features

        )



        train_end = int(

            total_samples *

            TRAIN_RATIO

        )



        validation_end = (

            train_end

            +

            int(

                total_samples *

                VALIDATION_RATIO

            )

        )




        if self.split == "train":


            self.features = self.features[:train_end]

            self.targets = self.targets[:train_end]



        elif self.split == "validation":


            self.features = (

                self.features[
                    train_end:
                    validation_end
                ]

            )


            self.targets = (

                self.targets[
                    train_end:
                    validation_end
                ]

            )



        else:


            self.features = (

                self.features[
                    validation_end:
                ]

            )


            self.targets = (

                self.targets[
                    validation_end:
                ]

            )







    # ======================================================
    # PyTorch API
    # ======================================================


    def __len__(self):


        return len(

            self.features

        )





    def __getitem__(

        self,

        index: int

    ) -> Tuple[torch.Tensor, torch.Tensor]:


        return (

            self.features[index],

            self.targets[index]

        )






# ==========================================================
# Test
# ==========================================================


if __name__ == "__main__":


    print("="*70)

    print(
        "Dynamic Traffic Dataset Test"
    )

    print("="*70)



    for h in [3,5,8]:


        dataset = TrafficDataset(

            split="train",

            horizon=h

        )


        x,y = dataset[0]



        print()

        print(
            f"Horizon : {h}"
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