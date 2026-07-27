"""
Window Generator

Responsibilities
----------------
1. Generate temporal input windows.
2. Generate prediction targets.
3. Support configurable history length.
4. Support configurable prediction horizon.
5. Support configurable stride.

This module DOES NOT:
- load datasets
- preprocess data
- normalize features
- build graphs
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from configs.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    SLIDING_WINDOW_STRIDE,
    LOCATION_COLUMN,
    NODE_COLUMN,
)


class WindowGenerator:
    """Generate temporal windows for traffic forecasting."""

    def __init__(
        self,
        history_length: int = HISTORY_LENGTH,
        prediction_horizon: int = PREDICTION_HORIZON,
        stride: int = SLIDING_WINDOW_STRIDE,
    ) -> None:

        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate input/output windows.

        Returns
        -------
        X : ndarray
            Shape:
            (samples, history_length, num_features)

        y : ndarray
            Shape:
            (samples, prediction_horizon)
        """

        x_windows = []
        y_windows = []

        grouped = dataframe.groupby(
            [LOCATION_COLUMN, NODE_COLUMN],
            sort=False,
        )

        for _, group in grouped:

            group = group.reset_index(drop=True)

            feature_values = group[FEATURE_COLUMNS].to_numpy(
                dtype=np.float32
            )

            target_values = group[TARGET_COLUMN].to_numpy(
                dtype=np.float32
            )

            total_steps = len(group)

            max_start = (
                total_steps
                - self.history_length
                - self.prediction_horizon
                + 1
            )

            if max_start <= 0:
                continue

            for start in range(
                0,
                max_start,
                self.stride,
            ):

                end = start + self.history_length

                target_end = (
                    end + self.prediction_horizon
                )

                x_windows.append(
                    feature_values[start:end]
                )

                y_windows.append(
                    target_values[end:target_end]
                )

        return (
            np.asarray(x_windows, dtype=np.float32),
            np.asarray(y_windows, dtype=np.float32),
        )


if __name__ == "__main__":

    from proposed_model.data.loader import TrafficDataset
    from data.preprocessing import DataPreprocessor
    from data.scaler import FeatureScaler

    dataset = TrafficDataset()

    dataframe = dataset.load()

    dataframe = DataPreprocessor(
        dataframe
    ).process()

    dataframe = FeatureScaler().fit_transform(
        dataframe
    )

    generator = WindowGenerator()

    X, y = generator.generate(dataframe)

    print("=" * 60)
    print("Window Generator")
    print("=" * 60)

    print(f"Input Shape : {X.shape}")
    print(f"Target Shape: {y.shape}")