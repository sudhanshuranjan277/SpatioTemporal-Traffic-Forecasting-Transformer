"""
Traffic Impact Graph Generation

Generates:

1. Waiting Time Comparison
2. Queue Length Comparison
3. Spillback Event Comparison


Input:
model_traffic_impact_results.csv


Output:
outputs/traffic_impact_graphs/
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "comparison_models"
    /
    "final_analysis"
    /
    "outputs"
)


GRAPH_DIR = (
    OUTPUT_DIR
    /
    "traffic_impact_graphs"
)


GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)



CSV_FILE = (
    OUTPUT_DIR
    /
    "model_traffic_impact_results.csv"
)





# ======================================================
# Load Data
# ======================================================


def load_data():

    df = pd.read_csv(
        CSV_FILE
    )

    return df





# ======================================================
# Waiting Time Graph
# ======================================================


def plot_waiting_time(df):


    plt.figure(
        figsize=(10,6)
    )


    for model in df["Model"].unique():

        data = df[
            df["Model"] == model
        ]


        plt.plot(

            data["Horizon"],

            data["Waiting_Time"],

            marker="o",

            label=model

        )



    plt.title(
        "Waiting Time Comparison"
    )

    plt.xlabel(
        "Prediction Horizon"
    )

    plt.ylabel(
        "Waiting Time"
    )


    plt.legend()

    plt.grid(True)


    plt.tight_layout()


    plt.savefig(

        GRAPH_DIR
        /
        "waiting_time_comparison.png"

    )


    plt.close()







# ======================================================
# Queue Length Graph
# ======================================================


def plot_queue_length(df):


    plt.figure(
        figsize=(10,6)
    )



    for model in df["Model"].unique():

        data = df[
            df["Model"] == model
        ]


        plt.plot(

            data["Horizon"],

            data["Queue_Length"],

            marker="o",

            label=model

        )



    plt.title(
        "Queue Length Comparison"
    )


    plt.xlabel(
        "Prediction Horizon"
    )


    plt.ylabel(
        "Queue Length"
    )


    plt.legend()

    plt.grid(True)


    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR
        /
        "queue_length_comparison.png"

    )


    plt.close()







# ======================================================
# Spillback Graph
# ======================================================


def plot_spillback(df):


    summary = (

        df.groupby("Model")

        ["Spillback_Events"]

        .sum()

        .reset_index()

    )



    plt.figure(

        figsize=(10,6)

    )


    plt.bar(

        summary["Model"],

        summary["Spillback_Events"]

    )


    plt.title(

        "Spillback Event Comparison"

    )


    plt.xlabel(

        "Model"

    )


    plt.ylabel(

        "Spillback Events"

    )


    plt.xticks(

        rotation=45

    )


    plt.grid(

        axis="y"

    )


    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR
        /
        "spillback_event_comparison.png"

    )


    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Traffic Impact Graph Generation"
    )

    print("="*70)



    df = load_data()



    plot_waiting_time(df)

    plot_queue_length(df)

    plot_spillback(df)



    print()

    print(
        "✓ Graphs Generated"
    )


    print(
        GRAPH_DIR
    )





if __name__=="__main__":

    main()