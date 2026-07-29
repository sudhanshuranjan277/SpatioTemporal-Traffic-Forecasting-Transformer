"""
Traffic Metrics Evaluation

Purpose
-------
Generate traffic condition metrics from Location 1 dataset.

Input
-----
datasets/processed/location_1_dataset.csv


Output
------
proposed_model/outputs/comparison/traffic_metrics/

    queue_length.csv
    waiting_time.csv
    spillback.csv


Metrics:
    - Queue Length
    - Waiting Time
    - Spillback Condition


Note:
-----
This script is independent from model training/testing.
It does not modify:
    - dataset.py
    - trainer.py
    - test.py
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
# Load Dataset
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
# Queue Length Metrics
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



    output_file = (
        OUTPUT_DIR
        /
        "queue_length.csv"
    )



    queue_df.to_csv(

        output_file,

        index=False

    )


    print(
        "Saved:",
        output_file
    )





# ======================================================
# Waiting Time Metrics
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



    output_file = (
        OUTPUT_DIR
        /
        "waiting_time.csv"
    )



    waiting_df.to_csv(

        output_file,

        index=False

    )


    print(
        "Saved:",
        output_file
    )





# ======================================================
# Spillback Metrics
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



    # Spillback rule
    #
    # Occupancy > 80%
    # indicates congestion spillback risk


    spillback_df["spillback"] = (

        spillback_df[
            "downstream_occupancy"
        ]

        >

        0.80

    )



    output_file = (
        OUTPUT_DIR
        /
        "spillback.csv"
    )



    spillback_df.to_csv(

        output_file,

        index=False

    )


    print(
        "Saved:",
        output_file
    )





# ======================================================
# Summary
# ======================================================


def print_summary(df):


    print()

    print("="*70)

    print(
        "Traffic Metrics Summary - Location 1"
    )

    print("="*70)



    print(
        f"Total Records : {len(df)}"
    )



    print()

    print(
        "Average Queue Length :",
        round(
            df["queue_length"].mean(),
            3
        )
    )


    print(
        "Average Waiting Time :",
        round(
            df["waiting_time"].mean(),
            3
        )
    )


    print(
        "Average Occupancy :",
        round(
            df["downstream_occupancy"].mean(),
            3
        )
    )



    spillback_count = (

        df["downstream_occupancy"]

        >

        0.80

    ).sum()



    print(
        "Spillback Events :",
        spillback_count
    )



    print("="*70)





# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Traffic Metrics Evaluation"
    )

    print("="*70)



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    df = load_dataset()



    print_summary(
        df
    )



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
        "Traffic Metrics Evaluation Completed"
    )

    print("="*70)





if __name__ == "__main__":

    main()