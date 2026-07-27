"""
data/loader.py

Traffic Dataset Loader

Responsibilities
----------------
1. Load one or multiple processed datasets.
2. Validate dataset existence.
3. Validate required columns.
4. Sort observations.
5. Return a clean pandas DataFrame.

This module DOES NOT:
--------------------
- preprocess data
- scale features
- generate sliding windows
- convert to tensors
- create PyTorch datasets
"""


from __future__ import annotations


from pathlib import Path


import pandas as pd



# ==========================================================
# Correct Package Import
# ==========================================================

from proposed_model.configs.config import (
    DATASET_FILES,
    LOCATION_COLUMN,
    NODE_COLUMN,
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMN,
)





class TrafficDataLoader:
    """
    Loader for processed traffic datasets.
    """


    def __init__(
        self,
        dataset_paths=None,
    ):


        if dataset_paths is None:

            dataset_paths = DATASET_FILES



        self.dataset_paths = [

            Path(path)

            for path in dataset_paths

        ]



    # ==========================================================
    # Public
    # ==========================================================


    def load(self) -> pd.DataFrame:
        """
        Load every dataset and merge them.
        """


        dataframes = []



        for dataset_path in self.dataset_paths:


            self._validate_file(
                dataset_path
            )



            dataframe = pd.read_csv(
                dataset_path
            )



            self._validate_columns(

                dataframe,

                dataset_path,

            )



            dataframe = self._sort_dataframe(

                dataframe,

            )



            dataframes.append(
                dataframe
            )



        if len(dataframes) == 0:

            raise RuntimeError(

                "No datasets were loaded."

            )



        dataframe = pd.concat(

            dataframes,

            ignore_index=True,

        )



        return dataframe





    # ==========================================================
    # Validation
    # ==========================================================


    @staticmethod
    def _validate_file(
        dataset_path: Path,
    ) -> None:


        if not dataset_path.exists():


            raise FileNotFoundError(

                f"Dataset not found:\n{dataset_path}"

            )





    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        dataset_path: Path,
    ) -> None:


        missing_columns = [


            column


            for column in REQUIRED_COLUMNS


            if column not in dataframe.columns


        ]



        if missing_columns:


            raise ValueError(


                f"\nDataset : {dataset_path.name}"

                f"\nMissing Columns : {missing_columns}"


            )





    @staticmethod
    def _sort_dataframe(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:



        dataframe = dataframe.sort_values(


            by=[

                LOCATION_COLUMN,

                NODE_COLUMN,

                TIMESTAMP_COLUMN,

            ]


        )



        dataframe = dataframe.reset_index(

            drop=True,

        )



        return dataframe





    # ==========================================================
    # Utility
    # ==========================================================


    def summary(self) -> None:


        dataframe = self.load()



        print("=" * 70)

        print(
            "Traffic Dataset Summary"
        )

        print("=" * 70)



        print(

            f"Datasets           : {len(self.dataset_paths)}"

        )


        print(

            f"Locations          : "
            f"{dataframe[LOCATION_COLUMN].nunique()}"

        )


        print(

            f"Junctions          : "
            f"{dataframe[NODE_COLUMN].nunique()}"

        )


        print(

            f"Rows               : {len(dataframe):,}"

        )


        print(

            f"Columns            : {len(dataframe.columns)}"

        )


        print("=" * 70)





# ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    loader = TrafficDataLoader()



    dataframe = loader.load()



    loader.summary()



    print()



    print(
        "First Five Rows"
    )

    print("-" * 70)



    print(
        dataframe.head()
    )