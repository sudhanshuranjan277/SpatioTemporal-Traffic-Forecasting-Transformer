"""
Data Preprocessing Module

Responsibilities
----------------
1. Remove duplicate rows
2. Handle missing values
3. Convert data types
4. Encode categorical features
5. Sort data
6. Return cleaned dataframe

NOTE:
This module DOES NOT:
- Scale features
- Generate windows
- Build graphs
"""

from __future__ import annotations

import pandas as pd

from configs.config import (
    FEATURE_COLUMNS,
    LOCATION_COLUMN,
    NODE_COLUMN,
    TIMESTAMP_COLUMN,
)


class DataPreprocessor:
    """Preprocess traffic forecasting dataset."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def process(self) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.
        """

        self._remove_duplicates()

        self._convert_timestamp()

        self._sort_dataframe()

        self._handle_missing_values()

        self._encode_categorical()

        self._optimize_dtypes()

        return self.df

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _remove_duplicates(self) -> None:
        """Remove duplicate rows."""
        self.df.drop_duplicates(inplace=True)

    def _convert_timestamp(self) -> None:
        """Convert timestamp column to numeric if possible."""

        self.df[TIMESTAMP_COLUMN] = pd.to_numeric(
            self.df[TIMESTAMP_COLUMN],
            errors="coerce"
        )

    def _sort_dataframe(self) -> None:
        """Sort dataframe."""

        self.df.sort_values(
            by=[
                LOCATION_COLUMN,
                NODE_COLUMN,
                TIMESTAMP_COLUMN,
            ],
            inplace=True,
        )

        self.df.reset_index(drop=True, inplace=True)

    def _handle_missing_values(self) -> None:
        """
        Handle missing values.

        Current Strategy:
        - Forward Fill
        - Backward Fill
        """

        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

    def _encode_categorical(self) -> None:
        """
        Encode categorical columns.

        Currently:
        traffic_event_type
        """

        if "traffic_event_type" in self.df.columns:

            self.df["traffic_event_type"] = (
                self.df["traffic_event_type"]
                .astype("category")
                .cat.codes
            )

    def _optimize_dtypes(self) -> None:
        """Reduce dataframe memory usage."""

        for column in FEATURE_COLUMNS:

            if column not in self.df.columns:
                continue

            if pd.api.types.is_float_dtype(self.df[column]):
                self.df[column] = self.df[column].astype("float32")

            elif pd.api.types.is_integer_dtype(self.df[column]):
                self.df[column] = self.df[column].astype("int32")

    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print preprocessing summary."""

        print("=" * 60)
        print("Preprocessed Dataset")
        print("=" * 60)

        print(f"Rows      : {len(self.df)}")
        print(f"Columns   : {len(self.df.columns)}")

        print("\nMissing Values")

        print(self.df.isnull().sum())

        print("=" * 60)


if __name__ == "__main__":

    from proposed_model.data.loader import TrafficDataset

    dataset = TrafficDataset()

    dataframe = dataset.load()

    processor = DataPreprocessor(dataframe)

    dataframe = processor.process()

    processor.summary()

    print(dataframe.head())