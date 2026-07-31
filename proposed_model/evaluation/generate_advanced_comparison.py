"""
Advanced Multi-Horizon Traffic Forecasting Comparison

Generates:
1. RMSE trend across horizons
2. MAE trend across horizons
3. RMSE heatmap
4. Model improvement percentage

Input:
------
outputs/comparison/horizon_*min/metrics.csv

Output:
-------
outputs/final_results/
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]


COMPARISON_DIR = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
)


FINAL_DIR = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "final_results"
)


FINAL_DIR.mkdir(
    exist_ok=True
)



# ==================================================
# Configuration
# ==================================================

HORIZONS = {

    "3 min":
        COMPARISON_DIR /
        "horizon_3min" /
        "metrics.csv",


    "5 min":
        COMPARISON_DIR /
        "horizon_5min" /
        "metrics.csv",


    "10 min":
        COMPARISON_DIR /
        "horizon_10min" /
        "metrics.csv"

}



MODELS = [

    "TransGTR",
    "LSTM",
    "GRU",
    "GraphWaveNet"

]


COLORS = {

    "TransGTR": "steelblue",
    "LSTM": "orange",
    "GRU": "green",
    "GraphWaveNet": "red"

}



# ==================================================
# Load CSVs
# ==================================================

def load_all_metrics():

    records = []


    for horizon, file in HORIZONS.items():


        print(
            "Reading:",
            file
        )


        df = pd.read_csv(file)


        df["Horizon"] = horizon


        records.append(df)



    data = pd.concat(
        records,
        ignore_index=True
    )


    data = data[
        data["Model"].isin(MODELS)
    ]


    return data





# ==================================================
# Horizon Trend Graph
# ==================================================

def plot_trend(
        df,
        metric
):


    plt.figure(
        figsize=(9,6)
    )


    for model in MODELS:


        temp = df[
            df["Model"] == model
        ]


        plt.plot(

            temp["Horizon"],

            temp[metric],

            marker="o",

            linewidth=2,

            markersize=8,

            label=model,

            color=COLORS[model]

        )



    plt.title(

        f"{metric} Variation Across Forecast Horizons",

        fontsize=14

    )


    plt.xlabel(
        "Forecast Horizon"
    )


    plt.ylabel(
        metric
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()



    output = (

        FINAL_DIR

        /

        f"horizon_{metric.lower()}_trend.png"

    )


    plt.savefig(

        output,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "Saved:",
        output
    )





# ==================================================
# RMSE Heatmap
# ==================================================

def create_rmse_heatmap(df):


    table = df.pivot(

        index="Model",

        columns="Horizon",

        values="RMSE"

    )



    plt.figure(

        figsize=(8,5)

    )


    plt.imshow(
        table.values,
        aspect="auto"
    )


    plt.colorbar(
        label="RMSE"
    )


    plt.xticks(

        range(len(table.columns)),

        table.columns

    )


    plt.yticks(

        range(len(table.index)),

        table.index

    )



    for i in range(
        len(table.index)
    ):

        for j in range(
            len(table.columns)
        ):


            plt.text(

                j,

                i,

                f"{table.iloc[i,j]:.2f}",

                ha="center",

                va="center"

            )



    plt.title(
        "RMSE Heatmap Across Forecast Horizons"
    )


    plt.xlabel(
        "Horizon"
    )


    plt.ylabel(
        "Model"
    )


    plt.tight_layout()



    output = (

        FINAL_DIR

        /

        "rmse_heatmap.png"

    )


    plt.savefig(

        output,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        output
    )





# ==================================================
# Improvement Percentage
# ==================================================

def create_improvement_chart(df):


    avg = df.groupby(
        "Model"
    )["RMSE"].mean()



    proposed = avg["TransGTR"]



    improvement = {}



    for model in MODELS:


        if model != "TransGTR":


            improvement[model] = (

                (

                    avg[model]
                    -
                    proposed

                )

                /

                avg[model]

            ) * 100




    plt.figure(

        figsize=(8,5)

    )


    plt.bar(

        improvement.keys(),

        improvement.values()

    )


    plt.ylabel(
        "Improvement (%)"
    )


    plt.title(
        "TransGTR RMSE Improvement Over Baselines"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()



    output = (

        FINAL_DIR

        /

        "model_improvement.png"

    )


    plt.savefig(

        output,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        output
    )





# ==================================================
# Main
# ==================================================

def main():


    print("="*70)

    print(
        "Generating Advanced Comparison Graphs"
    )

    print("="*70)



    df = load_all_metrics()



    df.to_csv(

        FINAL_DIR /
        "advanced_comparison_results.csv",

        index=False

    )



    plot_trend(
        df,
        "RMSE"
    )


    plot_trend(
        df,
        "MAE"
    )


    create_rmse_heatmap(
        df
    )


    create_improvement_chart(
        df
    )



    print()

    print("="*70)

    print(
        "Completed Successfully"
    )

    print("="*70)





if __name__ == "__main__":

    main()