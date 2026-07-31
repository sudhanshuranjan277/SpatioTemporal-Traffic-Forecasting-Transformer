"""
GRU Graph Generator

Generates:

1. Horizon wise metric comparison
2. Training loss curves
3. Actual vs Predicted traffic flow graphs

Input:
    outputs/
        metrics/
        predictions/
        GRU_*_history.json

Output:
    outputs/
        graphs/
"""


from pathlib import Path
import json


import torch
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


METRIC_DIR = OUTPUT_DIR / "metrics"


PREDICTION_DIR = OUTPUT_DIR / "predictions"


GRAPH_DIR = OUTPUT_DIR / "graphs"



HORIZONS = [3, 5, 8]



GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)





# ======================================================
# Load Metrics
# ======================================================


def load_metrics():

    results = {}


    for h in HORIZONS:


        file = (
            METRIC_DIR /
            f"GRU_{h}min_metrics.json"
        )


        with open(file, "r") as f:

            results[h] = json.load(f)



    return results





# ======================================================
# Metric Comparison Graph
# ======================================================


def generate_metric_graph():


    metrics = load_metrics()



    metric_names = [

        "MAE",

        "RMSE",

        "MAPE",

        "R2"

    ]



    for metric in metric_names:


        values = [

            metrics[h][metric]

            for h in HORIZONS

        ]



        plt.figure(figsize=(8,5))



        plt.plot(

            HORIZONS,

            values,

            marker="o"

        )



        plt.xlabel(
            "Forecast Horizon (Minutes)"
        )


        plt.ylabel(
            metric
        )


        plt.title(
            f"GRU {metric} Comparison"
        )


        plt.grid(True)



        plt.savefig(

            GRAPH_DIR /
            f"GRU_{metric}_comparison.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    print("✓ Metric graphs generated")







# ======================================================
# Prediction vs Actual
# ======================================================


def generate_prediction_graph():


    for h in HORIZONS:


        file = (

            PREDICTION_DIR /
            f"GRU_{h}min_predictions.pt"

        )


        data = torch.load(

            file,

            map_location="cpu"

        )


        prediction = data["prediction"]


        target = data["target"]



        # first sample
        # first junction


        predicted = (

            prediction[0,:,0]

            .numpy()

        )


        actual = (

            target[0,:,0]

            .numpy()

        )




        plt.figure(figsize=(8,5))



        plt.plot(

            actual,

            marker="o",

            label="Actual"

        )



        plt.plot(

            predicted,

            marker="x",

            label="Predicted"

        )



        plt.xlabel(
            "Future Time Step"
        )


        plt.ylabel(
            "Traffic Flow"
        )


        plt.title(

            f"GRU Actual vs Predicted ({h} min Horizon)"

        )


        plt.legend()


        plt.grid(True)



        plt.savefig(

            GRAPH_DIR /
            f"GRU_actual_vs_prediction_{h}min.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    print("✓ Prediction graphs generated")







# ======================================================
# Training Loss Graph
# ======================================================


def generate_loss_graph():


    for h in HORIZONS:


        history_file = (

            OUTPUT_DIR /
            f"GRU_{h}min_history.json"

        )



        if not history_file.exists():

            print(
                f"History missing for {h} min"
            )

            continue



        with open(history_file,"r") as f:

            history = json.load(f)





        plt.figure(figsize=(8,5))



        plt.plot(

            history["train_loss"],

            label="Train Loss"

        )



        plt.plot(

            history["val_loss"],

            label="Validation Loss"

        )



        plt.xlabel(
            "Epoch"
        )


        plt.ylabel(
            "Loss"
        )


        plt.title(

            f"GRU Training Loss ({h} min)"

        )


        plt.legend()


        plt.grid(True)



        plt.savefig(

            GRAPH_DIR /
            f"GRU_training_loss_{h}min.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    print("✓ Loss graphs generated")







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "GRU Graph Generation"
    )

    print("="*70)



    generate_metric_graph()


    generate_prediction_graph()


    generate_loss_graph()



    print()

    print(
        "Graphs saved at:"
    )


    print(
        GRAPH_DIR
    )


    print("="*70)





if __name__ == "__main__":

    main()