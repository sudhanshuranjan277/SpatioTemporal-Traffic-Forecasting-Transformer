"""
Sliding Window Generator

Responsibilities
----------------
1. Generate temporal history windows
2. Generate prediction targets
3. Preserve node dimension
4. Return model-ready tensors

Input
-----
Processed DataFrame

Output
------
X : (samples, history_length, num_nodes, num_features)

Y : (samples, prediction_horizon, num_nodes)
"""


from __future__ import annotations


import numpy as np
import pandas as pd



from proposed_model.configs.config import (
    FEATURE_COLUMNS,
    HISTORY_LENGTH,
    NODE_COLUMN,
    PREDICTION_HORIZON,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)





class WindowGenerator:


    def __init__(
        self,
        dataframe: pd.DataFrame,
        history: int = HISTORY_LENGTH,
        horizon: int = PREDICTION_HORIZON,
    ):


        self.df = dataframe.copy()

        self.history = history

        self.horizon = horizon





    def generate(self):


        # ---------------------------------------------------
        # Sort Data
        # ---------------------------------------------------

        self.df = self.df.sort_values(

            [
                TIMESTAMP_COLUMN,
                NODE_COLUMN,
            ]

        ).reset_index(drop=True)



        timestamps = sorted(

            self.df[TIMESTAMP_COLUMN].unique()

        )



        nodes = sorted(

            self.df[NODE_COLUMN].unique()

        )



        feature_tensor = []

        target_tensor = []



        # ---------------------------------------------------
        # Create temporal tensors
        # ---------------------------------------------------

        for timestamp in timestamps:


            frame = (

                self.df[

                    self.df[TIMESTAMP_COLUMN] == timestamp

                ]

                .set_index(NODE_COLUMN)

                .reindex(nodes)

            )



            # Handle missing nodes

            frame = frame.fillna(0)



            feature_tensor.append(

                frame[FEATURE_COLUMNS]

                .to_numpy(

                    dtype=np.float32

                )

            )



            target_tensor.append(

                frame[TARGET_COLUMN]

                .to_numpy(

                    dtype=np.float32

                )

            )





        feature_tensor = np.asarray(

            feature_tensor,

            dtype=np.float32,

        )



        target_tensor = np.asarray(

            target_tensor,

            dtype=np.float32,

        )





        # ---------------------------------------------------
        # Sliding Window
        # ---------------------------------------------------


        X = []

        Y = []



        total_time = feature_tensor.shape[0]



        end = (

            total_time

            -

            self.history

            -

            self.horizon

            +

            1

        )



        for start in range(end):


            X.append(

                feature_tensor[

                    start:

                    start + self.history

                ]

            )



            Y.append(

                target_tensor[

                    start + self.history:

                    start + self.history + self.horizon

                ]

            )





        X = np.asarray(

            X,

            dtype=np.float32,

        )



        Y = np.asarray(

            Y,

            dtype=np.float32,

        )



        return X, Y





    def summary(self):


        X, Y = self.generate()



        print("=" * 60)

        print(

            "Window Summary"

        )

        print("=" * 60)



        print(

            "Input Shape :",

            X.shape

        )



        print(

            "Target Shape:",

            Y.shape

        )



        print("=" * 60)





# ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    from proposed_model.data.loader import TrafficDataLoader



    print("=" * 60)

    print(
        "Window Generator Test"
    )

    print("=" * 60)



    loader = TrafficDataLoader()



    dataframe = loader.load()



    generator = WindowGenerator(

        dataframe

    )



    X, Y = generator.generate()



    generator.summary()



    print()

    print(
        "✓ Window generation successful"
    )