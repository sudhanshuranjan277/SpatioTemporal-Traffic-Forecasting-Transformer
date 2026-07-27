"""
Feature Scaling Module

Responsibilities
----------------
1. Fit scaler on training data only.
2. Transform train/validation/test datasets.
3. Save fitted scaler.
4. Load scaler during inference.

Author:
    Sudhanshu Ranjan

Project:
    Traffic Forecasting Research (TransGTR)
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

    
from configs.config import (
    FEATURE_COLUMNS,
    SCALER_SAVE_NAME,
    SCALER_TYPE,
)


class FeatureScaler:

    def __init__(
        self,
        save_directory: Path,
        scaler_type: str = SCALER_TYPE,
    ):

        self.save_directory = Path(save_directory)

        self.save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.scaler_path = (
            self.save_directory /
            SCALER_SAVE_NAME
        )

        scaler_type = scaler_type.lower()

        if scaler_type == "standard":
            self.scaler = StandardScaler()

        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()

        else:
            raise ValueError(
                f"Unsupported scaler type: {scaler_type}"
            )

    # ------------------------------------------------------

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Fit scaler using training dataframe only.
        """

        self.scaler.fit(
            dataframe[FEATURE_COLUMNS]
        )

    # ------------------------------------------------------

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform dataframe.
        """

        transformed = dataframe.copy()

        transformed[FEATURE_COLUMNS] = (
            self.scaler.transform(
                transformed[FEATURE_COLUMNS]
            )
        )

        return transformed

    # ------------------------------------------------------

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)

    # ------------------------------------------------------

    def inverse_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        restored = dataframe.copy()

        restored[FEATURE_COLUMNS] = (
            self.scaler.inverse_transform(
                restored[FEATURE_COLUMNS]
            )
        )

        return restored

    # ------------------------------------------------------

    def save(self) -> None:

        joblib.dump(
            self.scaler,
            self.scaler_path,
        )

    # ------------------------------------------------------

    def load(self) -> None:

        if not self.scaler_path.exists():

            raise FileNotFoundError(
                self.scaler_path
            )

        self.scaler = joblib.load(
            self.scaler_path
        )

    # ------------------------------------------------------

    @property
    def is_fitted(self) -> bool:

        return hasattr(
            self.scaler,
            "n_features_in_",
        )