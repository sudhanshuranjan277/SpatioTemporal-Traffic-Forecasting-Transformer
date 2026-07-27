"""
visualization/loss_curve.py

Training Loss Visualization

Input:
outputs/training_history.json

Output:
outputs/loss_curve.png
"""


from __future__ import annotations


import json

from pathlib import Path


import matplotlib.pyplot as plt


from configs.config import OUTPUT_DIR




def plot_loss_curve():


    history_file = (
        OUTPUT_DIR /
        "training_history.json"
    )


    if not history_file.exists():

        raise FileNotFoundError(
            f"Training history not found: {history_file}"
        )



    with open(
        history_file,
        "r"
    ) as file:

        history = json.load(file)



    train_loss = history["train_loss"]

    validation_loss = history["validation_loss"]



    plt.figure(
        figsize=(10,6)
    )


    plt.plot(
        train_loss,
        label="Training Loss"
    )


    plt.plot(
        validation_loss,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "TransGTR Training Loss"
    )


    plt.legend()


    plt.grid()



    save_path = (
        OUTPUT_DIR /
        "loss_curve.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    print(
        f"Loss curve saved: {save_path}"
    )




if __name__ == "__main__":

    plot_loss_curve()