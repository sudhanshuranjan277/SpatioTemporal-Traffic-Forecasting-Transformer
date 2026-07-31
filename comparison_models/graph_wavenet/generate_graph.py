"""
Graph WaveNet Graph Generator

Generates:

1. Metric comparison graphs
2. Actual vs Prediction graphs
3. Training loss curves


Output:

outputs/graphs/
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



GRAPH_DIR = (

    OUTPUT_DIR /

    "graphs"

)


METRIC_DIR = (

    OUTPUT_DIR /

    "metrics"

)


PREDICTION_DIR = (

    OUTPUT_DIR /

    "predictions"

)





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

        "MAE":[],
        "RMSE":[],
        "MAPE":[],
        "R2":[]

    }



    for h in HORIZONS:


        file = (

            METRIC_DIR /

            f"GraphWaveNet_{h}min_metrics.json"

        )


        if file.exists():


            with open(file) as f:


                data=json.load(f)



            for key in results:


                results[key].append(

                    data[key]

                )



    return results







# ======================================================
# Metric Graphs
# ======================================================


def generate_metric_graphs():



    metrics = load_metrics()



    for metric, values in metrics.items():



        plt.figure(

            figsize=(7,5)

        )



        plt.plot(

            HORIZONS,

            values,

            marker="o"

        )



        plt.xlabel(

            "Forecast Horizon (minutes)"

        )


        plt.ylabel(

            metric

        )


        plt.title(

            f"Graph WaveNet {metric} Comparison"

        )


        plt.grid()



        plt.savefig(

            GRAPH_DIR /

            f"GraphWaveNet_{metric}_comparison.png",

            dpi=300,

            bbox_inches="tight"

        )



        plt.close()







# ======================================================
# Prediction Graphs
# ======================================================


def generate_prediction_graphs():



    for h in HORIZONS:



        file = (

            PREDICTION_DIR /

            f"GraphWaveNet_{h}min_predictions.pt"

        )



        if not file.exists():

            continue



        data = torch.load(

            file,

            map_location="cpu"

        )



        prediction = data["prediction"]

        target = data["target"]



        prediction = prediction.flatten().numpy()

        target = target.flatten().numpy()



        plt.figure(

            figsize=(10,5)

        )



        plt.plot(

            target[:100],

            label="Actual"

        )


        plt.plot(

            prediction[:100],

            label="Predicted"

        )



        plt.xlabel(

            "Samples"

        )


        plt.ylabel(

            "Traffic Flow"

        )


        plt.title(

            f"Graph WaveNet Actual vs Prediction {h} min"

        )


        plt.legend()


        plt.grid()



        plt.savefig(

            GRAPH_DIR /

            f"GraphWaveNet_actual_vs_prediction_{h}min.png",

            dpi=300,

            bbox_inches="tight"

        )



        plt.close()







# ======================================================
# Loss Graphs
# ======================================================


def generate_loss_graphs():



    for h in HORIZONS:



        file = (

            OUTPUT_DIR /

            f"GraphWaveNet_{h}min_history.json"

        )



        if not file.exists():

            continue



        with open(file) as f:


            history=json.load(f)




        plt.figure(

            figsize=(7,5)

        )



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

            f"Graph WaveNet Training Loss {h} min"

        )


        plt.legend()

        plt.grid()



        plt.savefig(

            GRAPH_DIR /

            f"GraphWaveNet_training_loss_{h}min.png",

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

        "Graph WaveNet Graph Generation"

    )

    print("="*70)



    generate_metric_graphs()



    print(

        "✓ Metric graphs generated"

    )



    generate_prediction_graphs()



    print(

        "✓ Prediction graphs generated"

    )



    generate_loss_graphs()



    print(

        "✓ Loss graphs generated"

    )



    print()

    print(

        "Graphs saved at:"

    )

    print(

        GRAPH_DIR

    )



    print("="*70)






if __name__=="__main__":

    main()