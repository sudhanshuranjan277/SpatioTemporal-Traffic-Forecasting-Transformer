"""
Horizon Wise Model Performance Analysis

Generates:
1. RMSE comparison across horizons
2. MAE comparison across horizons
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



BASE_DIR = Path(__file__).resolve().parents[2]


OUTPUT_DIR = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
)


RESULT_DIR = (
    BASE_DIR
    /
    "proposed_model"
    /
    "outputs"
    /
    "final_results"
)


RESULT_DIR.mkdir(
    exist_ok=True
)



HORIZONS = {

    "3 min":
        OUTPUT_DIR / "horizon_3min" / "metrics.csv",

    "5 min":
        OUTPUT_DIR / "horizon_5min" / "metrics.csv",

    "10 min":
        OUTPUT_DIR / "horizon_10min" / "metrics.csv"

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



def load_results():


    records = []


    for horizon, file in HORIZONS.items():

        print("Reading:", file)


        df = pd.read_csv(file)


        df["Horizon"] = horizon


        records.append(df)



    return pd.concat(
        records,
        ignore_index=True
    )




def plot_metric(
        df,
        metric
):


    plt.figure(
        figsize=(9,6)
    )


    for model in MODELS:


        model_df = df[
            df["Model"] == model
        ]


        plt.plot(

            model_df["Horizon"],

            model_df[metric],

            marker="o",

            linewidth=2,

            label=model,

            color=COLORS[model]

        )


    plt.title(
        f"{metric} Performance Across Forecast Horizons"
    )


    plt.xlabel(
        "Forecast Horizon"
    )


    plt.ylabel(
        metric
    )


    plt.legend()

    plt.grid(True)


    plt.tight_layout()


    output = (
        RESULT_DIR
        /
        f"horizon_{metric.lower()}_comparison.png"
    )


    plt.savefig(
        output,
        dpi=300
    )


    plt.close()


    print(
        "Saved:",
        output
    )




def main():


    print("="*70)

    print(
        "Generating Horizon Analysis"
    )

    print("="*70)



    df = load_results()



    df.to_csv(

        RESULT_DIR
        /
        "horizon_comparison_results.csv",

        index=False

    )



    plot_metric(
        df,
        "RMSE"
    )


    plot_metric(
        df,
        "MAE"
    )


    print(
        "Completed"
    )




if __name__ == "__main__":

    main()