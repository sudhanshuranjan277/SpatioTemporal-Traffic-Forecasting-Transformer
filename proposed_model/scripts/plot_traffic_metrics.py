"""
Traffic Metrics Visualization

Input:

proposed_model/outputs/comparison/traffic_metrics/

    queue_length.csv
    waiting_time.csv
    spillback.csv


Output:

proposed_model/outputs/comparison/traffic_metrics/

    queue_length.png
    waiting_time.png
    spillback.png
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt




# ======================================================
# Paths
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRAFFIC_METRICS_DIR = (

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
# Plot Queue Length
# ======================================================


def plot_queue_length():


    file = (

        TRAFFIC_METRICS_DIR
        /
        "queue_length.csv"

    )


    df = pd.read_csv(
        file
    )



    plt.figure(
        figsize=(10,5)
    )



    plt.plot(

        df["simulation_time"],

        df["queue_length"],

        label="Queue Length"

    )



    plt.plot(

        df["simulation_time"],

        df["downstream_queue_length"],

        label="Downstream Queue"

    )



    plt.title(
        "Queue Length Analysis - Location 1"
    )


    plt.xlabel(
        "Simulation Time"
    )


    plt.ylabel(
        "Vehicles"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()



    output = (

        TRAFFIC_METRICS_DIR
        /
        "queue_length.png"

    )



    plt.savefig(

        output,

        dpi=300

    )


    plt.close()



    print(
        "Saved:",
        output
    )





# ======================================================
# Plot Waiting Time
# ======================================================


def plot_waiting_time():


    file = (

        TRAFFIC_METRICS_DIR
        /
        "waiting_time.csv"

    )


    df = pd.read_csv(
        file
    )



    plt.figure(

        figsize=(10,5)

    )



    plt.plot(

        df["simulation_time"],

        df["waiting_time"],

        label="Waiting Time"

    )



    plt.plot(

        df["simulation_time"],

        df["travel_time"],

        label="Travel Time"

    )



    plt.title(

        "Waiting Time Analysis - Location 1"

    )



    plt.xlabel(

        "Simulation Time"

    )


    plt.ylabel(

        "Time"

    )


    plt.legend()



    plt.grid(

        alpha=0.3

    )



    plt.tight_layout()



    output = (

        TRAFFIC_METRICS_DIR
        /
        "waiting_time.png"

    )



    plt.savefig(

        output,

        dpi=300

    )



    plt.close()



    print(

        "Saved:",

        output

    )





# ======================================================
# Plot Spillback
# ======================================================


def plot_spillback():


    file = (

        TRAFFIC_METRICS_DIR
        /
        "spillback.csv"

    )



    df = pd.read_csv(
        file
    )



    plt.figure(

        figsize=(10,5)

    )



    plt.plot(

        df["simulation_time"],

        df["downstream_occupancy"],

        label="Downstream Occupancy"

    )



    spillback_points = df[

        df["spillback"] == True

    ]



    if len(spillback_points) > 0:


        plt.scatter(

            spillback_points["simulation_time"],

            spillback_points["downstream_occupancy"],

            label="Spillback Event"

        )



    plt.axhline(

        y=0.80,

        linestyle="--",

        label="Spillback Threshold"

    )



    plt.title(

        "Spillback Analysis - Location 1"

    )



    plt.xlabel(

        "Simulation Time"

    )



    plt.ylabel(

        "Occupancy"

    )



    plt.legend()



    plt.grid(

        alpha=0.3

    )



    plt.tight_layout()



    output = (

        TRAFFIC_METRICS_DIR
        /
        "spillback.png"

    )



    plt.savefig(

        output,

        dpi=300

    )



    plt.close()



    print(

        "Saved:",

        output

    )





# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Traffic Metrics Visualization"
    )

    print("="*70)



    if not TRAFFIC_METRICS_DIR.exists():

        raise FileNotFoundError(

            "Traffic metrics folder not found"

        )



    plot_queue_length()


    plot_waiting_time()


    plot_spillback()



    print("="*70)

    print(
        "Traffic Metric Graphs Generated Successfully"
    )

    print("="*70)





if __name__ == "__main__":

    main()