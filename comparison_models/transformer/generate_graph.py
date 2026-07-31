"""
Transformer Graph Generator

Generates:

1. Actual vs Predicted
2. MAE comparison
3. RMSE comparison
4. R2 comparison
5. Training Loss


Supports:

3 min
5 min
8 min

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


HISTORY_DIR = OUTPUT_DIR / "history"


GRAPH_DIR = OUTPUT_DIR / "graphs"



GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)





HORIZONS = [

    3,

    5,

    8

]








# ======================================================
# Actual vs Prediction
# ======================================================


def actual_prediction_graph(horizon):


    file = (

        PREDICTION_DIR

        /

        f"Transformer_{horizon}min_predictions.pt"

    )



    data = torch.load(

        file,

        map_location="cpu"

    )



    prediction = data["prediction"]


    target = data["target"]





    prediction = prediction.flatten().numpy()


    target = target.flatten().numpy()




    plt.figure(

        figsize=(10,6)

    )



    plt.plot(

        target,

        label="Actual"

    )


    plt.plot(

        prediction,

        label="Predicted"

    )



    plt.xlabel(

        "Samples"

    )


    plt.ylabel(

        "Traffic Flow"

    )


    plt.title(

        f"Transformer Actual vs Prediction ({horizon} min)"

    )


    plt.legend()


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        f"Transformer_actual_vs_prediction_{horizon}min.png",

        dpi=300

    )



    plt.close()







# ======================================================
# Metric Comparison
# ======================================================


def metric_graph():



    metrics = {

        "MAE": [],

        "RMSE": [],

        "R2": []

    }



    horizons=[]




    for h in HORIZONS:



        file = (

            METRIC_DIR

            /

            f"Transformer_{h}min_metrics.json"

        )



        with open(file,"r") as f:


            data=json.load(f)



        horizons.append(

            f"{h} min"

        )


        metrics["MAE"].append(

            data["MAE"]

        )


        metrics["RMSE"].append(

            data["RMSE"]

        )


        metrics["R2"].append(

            data["R2"]

        )







    for name,values in metrics.items():



        plt.figure(

            figsize=(8,5)

        )



        plt.plot(

            horizons,

            values,

            marker="o"

        )


        plt.xlabel(

            "Forecast Horizon"

        )


        plt.ylabel(

            name

        )


        plt.title(

            f"Transformer {name} Comparison"

        )


        plt.grid()



        plt.tight_layout()



        plt.savefig(

            GRAPH_DIR /

            f"Transformer_{name}_comparison.png",

            dpi=300

        )



        plt.close()







# ======================================================
# Training Loss Graph
# ======================================================


def training_loss_graph(horizon):



    file = (

        HISTORY_DIR

        /

        f"Transformer_{horizon}min_history.json"

    )



    if not file.exists():

        return




    with open(file,"r") as f:


        history=json.load(f)





    plt.figure(

        figsize=(8,5)

    )



    plt.plot(

        history["train_loss"],

        label="Training Loss"

    )



    plt.plot(

        history["validation_loss"],

        label="Validation Loss"

    )



    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Loss"

    )


    plt.title(

        f"Transformer Training Loss ({horizon} min)"

    )


    plt.legend()


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        f"Transformer_training_loss_{horizon}min.png",

        dpi=300

    )


    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Transformer Graph Generation"

    )

    print("="*70)





    for h in HORIZONS:


        print(

            f"Generating {h} min graphs..."

        )


        actual_prediction_graph(h)


        training_loss_graph(h)





    metric_graph()





    print()

    print(

        "✓ Transformer Graphs Generated"

    )


    print()

    print(

        "Saved At:"

    )


    print(

        GRAPH_DIR

    )


    print("="*70)






if __name__=="__main__":


    main()