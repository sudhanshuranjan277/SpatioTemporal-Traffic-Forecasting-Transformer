"""
data/dataset.py

PyTorch Dataset for Traffic Forecasting

Responsibilities
----------------
1. Load processed traffic data.
2. Generate sliding windows.
3. Split Train / Validation / Test.
4. Convert data into tensors.
5. Provide DataLoader compatible samples.
"""


from __future__ import annotations


from typing import Tuple


import numpy as np
import torch


from torch.utils.data import Dataset



# ==========================================================
# Correct Package Imports
# ==========================================================


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
    """
    PyTorch Dataset for Traffic Forecasting.

    Returns
    -------

    x:
        (History, Nodes, Features)

    y:
        (Prediction Horizon, Nodes)
    """



    VALID_SPLITS = (

        "train",

        "validation",

        "test",

    )



    def __init__(

        self,

        split: str = "train",

    ):


        super().__init__()



        if split not in self.VALID_SPLITS:

            raise ValueError(

                f"Invalid split: {split}. "

                f"Choose from {self.VALID_SPLITS}"

            )



        self.split = split



        # ==================================================
        # Load Data
        # ==================================================

        loader = TrafficDataLoader()


        dataframe = loader.load()



        # Remove missing values

        dataframe = dataframe.fillna(0)



        # ==================================================
        # Sliding Windows
        # ==================================================

        window_generator = WindowGenerator(

            dataframe

        )



        features, targets = (

            window_generator.generate()

        )



        # ==================================================
        # Validation
        # ==================================================

        if np.isnan(features).any():

            raise ValueError(

                "NaN found in feature windows"

            )


        if np.isnan(targets).any():

            raise ValueError(

                "NaN found in target windows"

            )



        if np.isinf(features).any():

            raise ValueError(

                "Inf found in feature windows"

            )


        if np.isinf(targets).any():

            raise ValueError(

                "Inf found in target windows"

            )



        # ==================================================
        # Convert Tensor
        # ==================================================

        self.features = torch.tensor(

            features,

            dtype=torch.float32,

        )


        self.targets = torch.tensor(

            targets,

            dtype=torch.float32,

        )



        # ==================================================
        # Dataset Split
        # ==================================================

        self._create_split()





    # ======================================================
    # Split Train Validation Test
    # ======================================================


    def _create_split(self):


        total_samples = len(

            self.features

        )



        train_end = int(

            total_samples * TRAIN_RATIO

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


            self.features = (

                self.features[:train_end]

            )


            self.targets = (

                self.targets[:train_end]

            )



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
    # PyTorch Dataset API
    # ======================================================


    def __len__(self):

        return len(

            self.features

        )





    def __getitem__(

        self,

        index: int,

    ) -> Tuple[torch.Tensor, torch.Tensor]:


        x = self.features[index]


        y = self.targets[index]



        return x, y





# ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    print("=" * 70)

    print(

        "TrafficDataset Test"

    )

    print("=" * 70)



    train_dataset = TrafficDataset(

        split="train"

    )


    validation_dataset = TrafficDataset(

        split="validation"

    )


    test_dataset = TrafficDataset(

        split="test"

    )



    print(

        f"Train Samples      : {len(train_dataset)}"

    )


    print(

        f"Validation Samples : {len(validation_dataset)}"

    )


    print(

        f"Test Samples       : {len(test_dataset)}"

    )



    x, y = train_dataset[0]



    print("-" * 70)



    print(

        "Input Shape :",

        x.shape

    )


    print(

        "Target Shape:",

        y.shape

    )


    print(

        "Input Type:",

        type(x)

    )


    print(

        "Target Type:",

        type(y)

    )



    assert x.ndim == 3


    assert y.ndim == 2



    print("-" * 70)

    print(

        "✓ Dataset working correctly"

    )

    print("=" * 70)