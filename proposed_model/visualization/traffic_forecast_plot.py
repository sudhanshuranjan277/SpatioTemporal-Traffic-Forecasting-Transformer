"""
visualization/traffic_forecast_plot.py

Multi Junction Traffic Forecast Visualization

Input
-----
outputs/test_predictions.pt


Output
------
outputs/traffic_forecast.png
"""


from __future__ import annotations


import torch

import matplotlib.pyplot as plt


from configs.config import OUTPUT_DIR





# ==========================================================
# Load Prediction Data
# ==========================================================


def load_predictions():

    prediction_file = (
        OUTPUT_DIR /
        "test_predictions.pt"
    )


    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )


    data = torch.load(
        prediction_file,
        map_location="cpu",
    )


    predictions = data["predictions"]

    targets = data["targets"]


    return (
        predictions,
        targets,
    )





# ==========================================================
# Multi Node Forecast Plot
# ==========================================================


def plot_multi_node_forecast(

    predictions,

    targets,

    sample_index=0,

):


    if sample_index >= predictions.shape[0]:

        raise ValueError(
            "Invalid sample index"
        )



    prediction_sample = (
        predictions[
            sample_index
        ]
    )


    target_sample = (
        targets[
            sample_index
        ]
    )


    # Shape:
    #
    # (H,N)
    #


    horizon = range(
        prediction_sample.shape[0]
    )


    num_nodes = (
        prediction_sample.shape[1]
    )
    
    
        # ======================================================
    # Plot each junction
    # ======================================================


    plt.figure(
        figsize=(12, 7)
    )


    for node in range(num_nodes):


        actual = (
            target_sample[:, node]
            .numpy()
        )


        predicted = (
            prediction_sample[:, node]
            .numpy()
        )


        plt.plot(

            horizon,

            actual,

            marker="o",

            label=f"Node {node} Actual"

        )


        plt.plot(

            horizon,

            predicted,

            marker="x",

            linestyle="--",

            label=f"Node {node} Pred"

        )



    plt.xlabel(
        "Prediction Horizon"
    )


    plt.ylabel(
        "Traffic Flow"
    )


    plt.title(
        "Multi Junction Traffic Forecast"
    )


    plt.legend(
        fontsize=8,
        ncol=2,
    )


    plt.grid()



    save_path = (
        OUTPUT_DIR /
        "traffic_forecast.png"
    )



    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        f"Traffic forecast plot saved: {save_path}"
    )





# ==========================================================
# Main
# ==========================================================


def main():


    print("=" * 70)

    print(
        "Traffic Forecast Visualization"
    )

    print("=" * 70)



    predictions, targets = load_predictions()



    print(
        "Prediction Shape:",
        predictions.shape
    )


    print(
        "Target Shape:",
        targets.shape
    )



    plot_multi_node_forecast(

        predictions,

        targets,

        sample_index=0,

    )



    print("=" * 70)

    print(
        "Visualization Completed"
    )

    print("=" * 70)





if __name__ == "__main__":

    main()