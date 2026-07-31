"""
Generate Additional Metric Graphs

Reads:

final_metrics_updated.csv


Generates:

MSE comparison
Accuracy comparison

"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


INPUT_FILE = OUTPUT_DIR / "final_metrics_updated.csv"


GRAPH_DIR = OUTPUT_DIR / "graphs"



GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)







# ======================================================
# Model Colors
# ======================================================


MODEL_COLORS = {

    "LSTM": "blue",

    "GRU": "green",

    "GraphWaveNet": "orange",

    "Transformer": "red"

}







# ======================================================
# Metric Graph
# ======================================================


def create_graph(df, metric):


    plt.figure(

        figsize=(10,6)

    )




    for model in df["Model"].unique():


        data = df[

            df["Model"]

            ==

            model

        ]



        plt.plot(

            data["Horizon"],

            data[metric],

            marker="o",

            linewidth=2.5,

            markersize=7,

            color=MODEL_COLORS.get(

                model,

                "black"

            ),

            label=model

        )





    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        metric

    )


    plt.title(

        f"{metric} Comparison Across Models"

    )


    plt.legend()



    plt.grid(

        alpha=0.3

    )



    plt.tight_layout()




    plt.savefig(

        GRAPH_DIR /

        f"{metric}_all_models.png",

        dpi=300

    )



    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Additional Metric Graph Generation"

    )

    print("="*70)




    df = pd.read_csv(

        INPUT_FILE

    )





    metrics = [

        "MSE",

        "Accuracy"

    ]






    for metric in metrics:


        print(

            f"Generating {metric} graph..."

        )


        create_graph(

            df,

            metric

        )






    print()

    print(

        "✓ MSE and Accuracy graphs generated"

    )


    print()

    print(

        GRAPH_DIR

    )


    print("="*70)







if __name__=="__main__":

    main()