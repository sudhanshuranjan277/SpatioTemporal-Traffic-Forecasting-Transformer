"""
Radar Chart Model Performance Comparison

Purpose:
--------
Generate normalized radar chart for model comparison.

Models:
-------
TransGTR
LSTM
GRU
GraphWaveNet


Output:
-------
proposed_model/outputs/final_results/model_performance_radar.png
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]


INPUT_FILE = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "final_results"
    /
    "model_comparison.csv"
)



OUTPUT_FILE = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "final_results"
    /
    "model_performance_radar.png"
)



# ==================================================
# Models
# ==================================================

TARGET_MODELS = [

    "TransGTR",

    "LSTM",

    "GRU",

    "GraphWaveNet"

]



MODEL_COLORS = {

    "TransGTR": "steelblue",

    "LSTM": "orange",

    "GRU": "green",

    "GraphWaveNet": "red"

}



# ==================================================
# Load Data
# ==================================================

def load_data():


    print("=" * 70)

    print(
        "Loading Model Comparison Data"
    )

    print("=" * 70)



    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )



    df = pd.read_csv(
        INPUT_FILE
    )


    print()

    print(
        df
    )

    print()



    # Keep only required models

    df = df[
        df["Model"].isin(
            TARGET_MODELS
        )
    ]



    if len(df) == 0:

        raise ValueError(
            "No matching models found"
        )


    return df





# ==================================================
# Normalize Metrics
# ==================================================

def normalize_metrics(df):


    metrics = [

        "MSE",

        "MAE",

        "RMSE",

        "MAPE",

        "R2"

    ]


    normalized = df.copy()



    for metric in metrics:


        values = df[metric].astype(float)



        min_value = values.min()

        max_value = values.max()



        if max_value == min_value:


            normalized[metric] = 1.0



        else:


            if metric == "R2":


                # Higher is better

                normalized[metric] = (

                    values - min_value

                ) / (

                    max_value - min_value

                )



            else:


                # Lower is better

                normalized[metric] = (

                    max_value - values

                ) / (

                    max_value - min_value

                )



    return normalized





# ==================================================
# Radar Plot
# ==================================================

def create_radar(df):


    metrics = [

        "MSE",

        "MAE",

        "RMSE",

        "MAPE",

        "R2"

    ]



    values_count = len(metrics)



    angles = np.linspace(

        0,

        2 * np.pi,

        values_count,

        endpoint=False

    )


    angles = np.concatenate(

        [

            angles,

            [angles[0]]

        ]

    )



    fig = plt.figure(

        figsize=(8,8)

    )


    ax = fig.add_subplot(

        111,

        polar=True

    )



    for _, row in df.iterrows():


        values = [

            row[m]

            for m in metrics

        ]


        values.append(

            values[0]

        )



        model = row["Model"]



        ax.plot(

            angles,

            values,

            linewidth=2,

            label=model,

            color=MODEL_COLORS.get(

                model,

                "gray"

            )

        )


        ax.fill(

            angles,

            values,

            alpha=0.1,

            color=MODEL_COLORS.get(

                model,

                "gray"

            )

        )




    ax.set_xticks(

        angles[:-1]

    )


    ax.set_xticklabels(

        metrics

    )


    ax.set_ylim(

        0,

        1

    )



    plt.title(

        "Normalized Model Performance Radar Comparison",

        fontsize=14

    )


    plt.legend(

        loc="upper right",

        bbox_to_anchor=(1.3,1.1)

    )


    plt.tight_layout()



    plt.savefig(

        OUTPUT_FILE,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print()

    print(
        "Saved:",
        OUTPUT_FILE
    )





# ==================================================
# Main
# ==================================================

def main():


    df = load_data()


    normalized_df = normalize_metrics(

        df

    )


    create_radar(

        normalized_df

    )


    print()

    print("=" * 70)

    print(
        "Radar Chart Generated Successfully"
    )

    print("=" * 70)





if __name__ == "__main__":

    main()