"""
Research Level Graph Generation

Generates:

1. MAE Comparison
2. RMSE Comparison
3. R2 Comparison
4. Accuracy Comparison
5. Horizon Wise RMSE

For:
LSTM
GRU
GraphWaveNet
Transformer

Input:
final_combined_results.csv

Output:
research_graphs/
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt




# ======================================================
# Paths
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "comparison_models"
    /
    "final_analysis"
    /
    "outputs"
)


CSV_FILE = (
    OUTPUT_DIR
    /
    "final_combined_results.csv"
)


GRAPH_DIR = (
    OUTPUT_DIR
    /
    "research_graphs"
)


GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)




# ======================================================
# Fixed Model Colors
# ======================================================


MODEL_COLORS = {

    "LSTM": "blue",

    "GRU": "orange",

    "GraphWaveNet": "green",

    "Transformer": "red"

}





# ======================================================
# Load Data
# ======================================================


def load_data():

    df = pd.read_csv(
        CSV_FILE
    )

    return df





# ======================================================
# Metric Bar Graph
# ======================================================


def create_metric_graph(
        df,
        metric,
        title,
        filename
):


    summary = (

        df.groupby("Model")[metric]

        .mean()

        .reset_index()

    )



    plt.figure(
        figsize=(10,6)
    )


    plt.bar(

        summary["Model"],

        summary[metric],

        color=[
            MODEL_COLORS[m]
            for m in summary["Model"]
        ]

    )



    plt.title(
        title
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


    plt.grid(
        axis="y"
    )


    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR
        /
        filename,

        dpi=300

    )


    plt.close()







# ======================================================
# Horizon Graph
# ======================================================


def create_horizon_rmse(df):


    plt.figure(
        figsize=(10,6)
    )



    for model in df["Model"].unique():


        data = df[
            df["Model"] == model
        ]



        plt.plot(

            data["Horizon"],

            data["RMSE"],

            marker="o",

            label=model,

            color=MODEL_COLORS[model]

        )



    plt.title(
        "RMSE Variation with Forecast Horizon"
    )


    plt.xlabel(
        "Forecast Horizon"
    )


    plt.ylabel(
        "RMSE"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR
        /
        "horizon_rmse_comparison.png",

        dpi=300

    )


    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(
        "Research Graph Generation"
    )

    print("="*70)



    df = load_data()



    create_metric_graph(

        df,

        "MAE",

        "Model Comparison - MAE",

        "mae_comparison.png"

    )



    create_metric_graph(

        df,

        "RMSE",

        "Model Comparison - RMSE",

        "rmse_comparison.png"

    )



    create_metric_graph(

        df,

        "R2",

        "Model Comparison - R2 Score",

        "r2_comparison.png"

    )



    create_metric_graph(

        df,

        "Accuracy",

        "Model Accuracy Comparison",

        "accuracy_comparison.png"

    )



    create_horizon_rmse(
        df
    )



    print()

    print(
        "✓ Research Graphs Generated"
    )


    print(
        GRAPH_DIR
    )





if __name__ == "__main__":

    main()