"""
Traffic Metrics Generator

Dataset:
    datasets/processed/location_1_dataset.csv


Generated Output:

proposed_model/outputs/comparison/traffic_metrics/

    queue_length.csv
    waiting_time.csv
    spillback.csv

"""


from pathlib import Path

import pandas as pd




# ======================================================
# Project Paths
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]



DATASET_PATH = (

    PROJECT_ROOT
    /
    "datasets"
    /
    "processed"
    /
    "location_1_dataset.csv"

)



OUTPUT_DIR = (

    PROJECT_ROOT
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
    /
    "traffic_metrics"

)





# ======================================================
# Load Location 1 Dataset
# ======================================================


def load_dataset():


    if not DATASET_PATH.exists():

        raise FileNotFoundError(

            f"Dataset not found:\n{DATASET_PATH}"

        )



    print(
        "Loading Dataset:"
    )

    print(
        DATASET_PATH
    )



    df = pd.read_csv(
        DATASET_PATH
    )



    return df





# ======================================================
# Generate Queue Length Metrics
# ======================================================


def generate_queue_metrics(df):


    queue_df = df[

        [

            "simulation_time",

            "location_id",

            "junction_id",

            "queue_length",

            "downstream_queue_length"

        ]

    ]



    queue_path = (

        OUTPUT_DIR
        /
        "queue_length.csv"

    )



    queue_df.to_csv(

        queue_path,

        index=False

    )



    print(
        "Saved:",
        queue_path
    )





# ======================================================
# Generate Waiting Time Metrics
# ======================================================


def generate_waiting_metrics(df):


    waiting_df = df[

        [

            "simulation_time",

            "location_id",

            "junction_id",

            "waiting_time",

            "travel_time"

        ]

    ]



    waiting_path = (

        OUTPUT_DIR
        /
        "waiting_time.csv"

    )



    waiting_df.to_csv(

        waiting_path,

        index=False

    )



    print(
        "Saved:",
        waiting_path
    )





# ======================================================
# Generate Spillback Metrics
# ======================================================


def generate_spillback_metrics(df):


    spillback_df = df[

        [

            "simulation_time",

            "location_id",

            "junction_id",

            "downstream_occupancy",

            "downstream_queue_length"

        ]

    ].copy()



    # Spillback condition
    #
    # occupancy > 80%
    # means downstream congestion risk


    spillback_df["spillback"] = (

        spillback_df["downstream_occupancy"]

        >

        0.80

    )



    spillback_path = (

        OUTPUT_DIR
        /
        "spillback.csv"

    )



    spillback_df.to_csv(

        spillback_path,

        index=False

    )



    print(
        "Saved:",
        spillback_path
    )





# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Traffic Metrics Generation - Location 1"
    )

    print("="*70)



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    df = load_dataset()



    print()

    print(
        "Total Records:",
        len(df)
    )


    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )



    print()



    generate_queue_metrics(
        df
    )



    generate_waiting_metrics(
        df
    )



    generate_spillback_metrics(
        df
    )



    print()

    print("="*70)

    print(
        "Traffic Metrics Generation Completed"
    )

    print("="*70)





if __name__ == "__main__":

    main()