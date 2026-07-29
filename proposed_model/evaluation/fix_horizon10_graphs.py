"""
Fix Horizon 10 Minutes Graph Generation

Purpose:
--------
Regenerate corrupted horizon_10min comparison graphs.

Input:
------
proposed_model/outputs/comparison/horizon_10min/metrics.csv

Output:
-------
mae.png
mse.png
rmse.png
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]


METRICS_FILE = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
    /
    "horizon_10min"
    /
    "metrics.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
    /
    "horizon_10min"
)



# ==================================================
# Model Colors
# ==================================================

MODEL_COLORS = {

    "TransGTR": "steelblue",

    "LSTM": "orange",

    "GRU": "green",

    "GraphWaveNet": "red"

}



# ==================================================
# Load Metrics
# ==================================================

def load_metrics():


    print("=" * 70)

    print(
        "Loading Horizon 10 Minutes Metrics"
    )

    print("=" * 70)



    if not METRICS_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {METRICS_FILE}"
        )


    df = pd.read_csv(
        METRICS_FILE
    )


    print()

    print(df)

    print()


    return df




# ==================================================
# Generate Graph
# ==================================================

def generate_graph(
        df,
        metric
):


    plt.figure(
        figsize=(8,5)
    )


    colors = [

        MODEL_COLORS.get(
            model,
            "gray"
        )

        for model in df["Model"]

    ]



    plt.bar(

        df["Model"],

        df[metric],

        color=colors

    )



    plt.title(

        f"{metric} Comparison - Horizon 10 Minutes",

        fontsize=14

    )


    plt.xlabel(
        "Model"
    )


    plt.ylabel(
        metric
    )


    plt.xticks(
        rotation=45
    )


    # Add values above bars

    for index, value in enumerate(df[metric]):


        plt.text(

            index,

            value,

            f"{value:.2f}",

            ha="center",

            va="bottom",

            fontsize=9

        )



    plt.tight_layout()



    output_file = (

        OUTPUT_DIR

        /

        f"{metric.lower()}.png"

    )



    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        output_file
    )





# ==================================================
# Main
# ==================================================

def main():


    df = load_metrics()



    metrics = [

        "MSE",

        "MAE",

        "RMSE"

    ]



    for metric in metrics:


        if metric in df.columns:


            generate_graph(

                df,

                metric

            )

        else:


            print(

                f"Skipping {metric}, column missing"

            )



    print()

    print("=" * 70)

    print(
        "Horizon 10 Minutes Graph Generation Completed"
    )

    print("=" * 70)





if __name__ == "__main__":

    main()