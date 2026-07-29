"""
Training Loss Curve Generator

Models:
    - TransGTR
    - LSTM
    - GRU
    - GraphWaveNet


Output:

proposed_model/outputs/comparison/loss_curves/

    transgtr_loss.png
    lstm_loss.png
    gru_loss.png
    gwnet_loss.png

"""


from pathlib import Path
import json

import matplotlib.pyplot as plt




# ======================================================
# Project Root
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]



# ======================================================
# Output Directory
# ======================================================


LOSS_OUTPUT_DIR = (

    PROJECT_ROOT
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
    /
    "loss_curves"

)




# ======================================================
# Training History Paths
# ======================================================


HISTORY_FILES = {


    "TransGTR":

    PROJECT_ROOT
    /
    "proposed_model"
    /
    "outputs"
    /
    "metrics"
    /
    "training_history.json",



    "LSTM":

    PROJECT_ROOT
    /
    "comparison_models"
    /
    "lstm"
    /
    "outputs"
    /
    "training_history.json",



    "GRU":

    PROJECT_ROOT
    /
    "comparison_models"
    /
    "gru"
    /
    "outputs"
    /
    "training_history.json",



    "GraphWaveNet":

    PROJECT_ROOT
    /
    "comparison_models"
    /
    "graph_wavenet"
    /
    "outputs"
    /
    "training_history.json"

}




# ======================================================
# Colors
# ======================================================


MODEL_COLORS = {


    "TransGTR":
    "red",


    "LSTM":
    "blue",


    "GRU":
    "green",


    "GraphWaveNet":
    "orange"

}




# ======================================================
# Load History
# ======================================================


def load_history(path):


    with open(

        path,

        "r"

    ) as file:


        return json.load(file)




# ======================================================
# Extract Loss
# ======================================================


def extract_losses(history):


    train_loss = (

        history.get(
            "train_loss"
        )

        or

        history.get(
            "training_loss"
        )

        or

        history.get(
            "loss"
        )

    )



    val_loss = (

        history.get(
            "val_loss"
        )

        or

        history.get(
            "validation_loss"
        )

        or

        history.get(
            "valid_loss"
        )

    )



    if train_loss is None:

        raise ValueError(
            "Training loss not found in history file"
        )



    if val_loss is None:

        val_loss = []



    return train_loss, val_loss




# ======================================================
# Generate Plot
# ======================================================


def generate_plot(

        model_name,

        history

):


    train_loss, val_loss = extract_losses(
        history
    )



    plt.figure(

        figsize=(9,5)

    )



    plt.plot(

        train_loss,

        label="Train Loss",

        color=MODEL_COLORS[model_name]

    )



    if len(val_loss) > 0:


        plt.plot(

            val_loss,

            label="Validation Loss",

            linestyle="--"

        )



    plt.title(

        f"{model_name} Training Loss"

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.legend()



    plt.grid(
        alpha=0.3
    )



    LOSS_OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    filename = (

        model_name.lower()

        .replace(

            "graphwavenet",

            "gwnet"

        )

        +

        "_loss.png"

    )



    save_path = (

        LOSS_OUTPUT_DIR
        /
        filename

    )



    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        save_path
    )




# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Generating Loss Curves"
    )

    print("="*70)



    for model_name, history_path in HISTORY_FILES.items():



        if not history_path.exists():


            print(
                "Missing:",
                history_path
            )

            continue




        print()

        print(
            "Processing:",
            model_name
        )



        history = load_history(

            history_path

        )



        generate_plot(

            model_name,

            history

        )



    print()

    print("="*70)

    print(
        "Loss Curve Generation Completed"
    )

    print("="*70)





if __name__ == "__main__":

    main()