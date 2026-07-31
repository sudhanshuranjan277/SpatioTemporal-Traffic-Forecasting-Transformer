"""
LSTM Visualization Generator

Generates:

1. Horizon vs MAE
2. Horizon vs RMSE
3. Horizon vs R2
4. Training Loss Curves
5. Prediction vs Actual Traffic Flow
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


GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)





HORIZONS = [3,5,8]





# ======================================================
# Load Metrics
# ======================================================


def load_metrics():


    results = {

        "horizon": [],

        "MAE": [],

        "RMSE": [],

        "R2": []

    }



    for h in HORIZONS:


        file = (

            METRIC_DIR

            /

            f"LSTM_{h}min_metrics.json"

        )


        with open(file,"r") as f:


            data = json.load(f)



        results["horizon"].append(h)

        results["MAE"].append(data["MAE"])

        results["RMSE"].append(data["RMSE"])

        results["R2"].append(data["R2"])



    return results






# ======================================================
# Horizon Graphs
# ======================================================


def plot_metric(

    x,

    y,

    ylabel,

    filename

):


    plt.figure(figsize=(7,5))


    plt.plot(

        x,

        y,

        marker="o"

    )


    plt.xlabel(

        "Forecast Horizon (minutes)"

    )


    plt.ylabel(ylabel)


    plt.title(

        f"LSTM {ylabel} vs Forecast Horizon"

    )


    plt.grid(True)



    plt.savefig(

        GRAPH_DIR / filename,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()






# ======================================================
# Training Loss
# ======================================================


def plot_training_loss():


    plt.figure(figsize=(8,5))



    for h in HORIZONS:


        file = (

            OUTPUT_DIR

            /

            f"LSTM_{h}min_history.json"

        )



        with open(file,"r") as f:


            history = json.load(f)



        plt.plot(

            history["train_loss"],

            label=f"{h} min"

        )



    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Training Loss"

    )


    plt.title(

        "LSTM Training Loss Comparison"

    )


    plt.legend()


    plt.grid(True)



    plt.savefig(

        GRAPH_DIR / "training_loss.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()






# ======================================================
# Prediction vs Actual
# ======================================================


def plot_prediction_actual():


    file = (

        PREDICTION_DIR

        /

        "LSTM_5min_predictions.pt"

    )



    data = torch.load(

        file,

        map_location="cpu"

    )



    prediction = data["prediction"]

    target = data["target"]




    # First test sample, first junction


    pred = prediction[0,:,0].numpy()

    actual = target[0,:,0].numpy()



    plt.figure(figsize=(8,5))


    plt.plot(

        actual,

        marker="o",

        label="Actual"

    )


    plt.plot(

        pred,

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

        "LSTM Prediction vs Actual (5 min Horizon)"

    )


    plt.legend()


    plt.grid(True)



    plt.savefig(

        GRAPH_DIR / "prediction_vs_actual.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()






# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Generating LSTM Graphs"

    )

    print("="*70)




    metrics = load_metrics()



    plot_metric(

        metrics["horizon"],

        metrics["MAE"],

        "MAE",

        "horizon_mae.png"

    )



    plot_metric(

        metrics["horizon"],

        metrics["RMSE"],

        "RMSE",

        "horizon_rmse.png"

    )



    plot_metric(

        metrics["horizon"],

        metrics["R2"],

        "R2 Score",

        "horizon_r2.png"

    )



    plot_training_loss()



    plot_prediction_actual()



    print()

    print(

        "Graphs saved at:"

    )


    print(GRAPH_DIR)


    print("="*70)





if __name__ == "__main__":

    main()