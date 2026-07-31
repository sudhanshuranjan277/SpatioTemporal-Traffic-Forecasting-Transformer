"""
Final Research Report Graph Generator


Reads:

final_metrics.csv
best_model_summary.csv
improvement_analysis.csv


Generates:

best_model_horizon.png
mae_improvement.png
overall_performance_heatmap.png

"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


REPORT_GRAPH_DIR = OUTPUT_DIR / "report_graphs"



REPORT_GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)





METRICS_FILE = OUTPUT_DIR / "final_metrics.csv"

BEST_FILE = OUTPUT_DIR / "best_model_summary.csv"

IMPROVEMENT_FILE = OUTPUT_DIR / "improvement_analysis.csv"







# ======================================================
# Model Colors
# ======================================================


MODEL_COLORS = {


    "LSTM":

        "blue",


    "GRU":

        "green",


    "GraphWaveNet":

        "orange",


    "Transformer":

        "red"

}







# ======================================================
# Load Data
# ======================================================


def load_data():


    metrics = pd.read_csv(

        METRICS_FILE

    )


    best = pd.read_csv(

        BEST_FILE

    )


    improvement = pd.read_csv(

        IMPROVEMENT_FILE

    )



    return metrics,best,improvement







# ======================================================
# Best Model Graph
# ======================================================


def best_model_graph(best):


    plt.figure(

        figsize=(8,5)

    )



    colors = [

        MODEL_COLORS.get(

            model,

            "black"

        )

        for model in best["Best Model"]

    ]





    plt.bar(

        best["Horizon"],

        best["Best Model"],

        color=colors

    )



    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        "Best Model"

    )


    plt.title(

        "Best Performing Model Across Forecast Horizons"

    )



    plt.grid(

        axis="y",

        alpha=0.3

    )



    plt.tight_layout()



    plt.savefig(

        REPORT_GRAPH_DIR /

        "best_model_horizon.png",

        dpi=300

    )


    plt.close()







# ======================================================
# MAE Improvement
# ======================================================


def improvement_graph(improvement):


    plt.figure(

        figsize=(8,5)

    )



    plt.bar(

        improvement["Horizon"],

        improvement["MAE Improvement %"],

        color="purple"

    )



    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        "MAE Improvement (%)"

    )



    plt.title(

        "MAE Improvement Compared With LSTM Baseline"

    )



    plt.grid(

        axis="y",

        alpha=0.3

    )



    plt.tight_layout()



    plt.savefig(

        REPORT_GRAPH_DIR /

        "mae_improvement.png",

        dpi=300

    )


    plt.close()







# ======================================================
# Heatmap
# ======================================================


def performance_heatmap(metrics):


    avg_metrics = metrics.groupby(

        "Model"

    )[

        [

            "MAE",

            "RMSE",

            "MAPE",

            "R2"

        ]

    ].mean()



    plt.figure(

        figsize=(8,5)

    )



    plt.imshow(

        avg_metrics.values,

        aspect="auto"

    )



    plt.xticks(

        range(len(avg_metrics.columns)),

        avg_metrics.columns

    )



    plt.yticks(

        range(len(avg_metrics.index)),

        avg_metrics.index

    )



    plt.colorbar()



    plt.title(

        "Overall Model Performance Heatmap"

    )



    plt.tight_layout()



    plt.savefig(

        REPORT_GRAPH_DIR /

        "overall_performance_heatmap.png",

        dpi=300

    )



    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Final Research Report Graph Generation"

    )

    print("="*70)





    metrics,best,improvement = load_data()





    print(

        "Generating best model graph..."

    )


    best_model_graph(best)





    print(

        "Generating MAE improvement graph..."

    )


    improvement_graph(improvement)





    print(

        "Generating performance heatmap..."

    )


    performance_heatmap(metrics)






    print()

    print(

        "✓ Report graphs generated"

    )


    print(

        REPORT_GRAPH_DIR

    )


    print("="*70)






if __name__=="__main__":

    main()