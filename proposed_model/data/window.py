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

from configs.config import (
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
        # Sort
        # ---------------------------------------------------

        self.df = self.df.sort_values(
            [TIMESTAMP_COLUMN, NODE_COLUMN]
        ).reset_index(drop=True)

        timestamps = sorted(
            self.df[TIMESTAMP_COLUMN].unique()
        )

        nodes = sorted(
            self.df[NODE_COLUMN].unique()
        )

        num_nodes = len(nodes)
        num_features = len(FEATURE_COLUMNS)

        feature_tensor = []
        target_tensor = []

        # ---------------------------------------------------
        # Build tensor per timestamp
        # ---------------------------------------------------

        for timestamp in timestamps:

            frame = (
                self.df[
                    self.df[TIMESTAMP_COLUMN] == timestamp
                ]
                .set_index(NODE_COLUMN)
                .reindex(nodes)
            )
            
            frame = frame.fillna(0)

            feature_tensor.append(
                frame[FEATURE_COLUMNS].to_numpy(
                    dtype=np.float32
                )
            )

            target_tensor.append(
                frame[TARGET_COLUMN].to_numpy(
                    dtype=np.float32
                )
            )

        feature_tensor = np.asarray(feature_tensor)

        target_tensor = np.asarray(target_tensor)

        # Shape:
        # feature_tensor -> (time, nodes, features)
        # target_tensor  -> (time, nodes)

        X = []
        Y = []

        total_time = feature_tensor.shape[0]

        end = (
            total_time
            - self.history
            - self.horizon
            + 1
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

        X = np.asarray(X, dtype=np.float32)

        Y = np.asarray(Y, dtype=np.float32)

        return X, Y

    def summary(self):

        X, Y = self.generate()

        print("=" * 60)
        print("Window Summary")
        print("=" * 60)

        print("Input Shape :", X.shape)
        print("Target Shape:", Y.shape)

        print("=" * 60)


if __name__ == "__main__":

    from proposed_model.data.loader import TrafficDataset
    from data.preprocessing import DataPreprocessor
    from data.scaler import FeatureScaler
    from pathlib import Path

    dataset = TrafficDataset()

    df = dataset.load()

    df = DataPreprocessor(df).process()

    df = FeatureScaler(
        save_directory=Path("checkpoints")
    ).fit_transform(df)

    generator = WindowGenerator(df)

    X, Y = generator.generate()

    generator.summary()