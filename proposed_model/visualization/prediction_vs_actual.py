"""
visualization/prediction_vs_actual.py

Actual vs Predicted Traffic Flow Visualization


Input
-----
outputs/test_predictions.pt


Output
------
outputs/prediction_vs_actual.png
"""


from __future__ import annotations


from pathlib import Path


import torch
import matplotlib.pyplot as plt


from configs.config import OUTPUT_DIR




# ==========================================================
# Load Predictions
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



    return predictions, targets





# ==========================================================
# Plot
# ==========================================================


def plot_prediction_vs_actual(

    predictions,

    targets,

    sample_index=0,

    node_index=0,

):


    if sample_index >= predictions.shape[0]:

        raise ValueError(
            "Invalid sample index"
        )


    if node_index >= predictions.shape[2]:

        raise ValueError(
            "Invalid node index"
        )



    predicted = (
        predictions[
            sample_index,
            :,
            node_index
        ]
        .numpy()
    )


    actual = (
        targets[
            sample_index,
            :,
            node_index
        ]
        .numpy()
    )



    horizon = range(
        len(actual)
    )



    plt.figure(
        figsize=(10,6)
    )


    plt.plot(
        horizon,
        actual,
        marker="o",
        label="Actual"
    )


    plt.plot(
        horizon,
        predicted,
        marker="x",
        label="Predicted"
    )



    plt.xlabel(
        "Prediction Horizon"
    )


    plt.ylabel(
        "Traffic Flow"
    )


    plt.title(
        f"Traffic Forecast - Node {node_index}"
    )


    plt.legend()


    plt.grid()



    save_path = (
        OUTPUT_DIR /
        "prediction_vs_actual.png"
    )



    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    print(
        f"Prediction plot saved: {save_path}"
    )
    
    # ==========================================================
# Main
# ==========================================================


def main():

    print("=" * 70)

    print(
        "Prediction vs Actual Visualization"
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



    plot_prediction_vs_actual(

        predictions,

        targets,

        sample_index=0,

        node_index=0,

    )


    print("=" * 70)

    print(
        "Visualization Completed"
    )

    print("=" * 70)




# ==========================================================
# Run
# ==========================================================


if __name__ == "__main__":

    main()
    