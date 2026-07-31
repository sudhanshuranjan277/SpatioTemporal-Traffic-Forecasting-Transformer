"""
Final Model Comparison Graph Generator

Models:

LSTM
GRU
GraphWaveNet
Transformer


Reads:

outputs/final_metrics.csv


Generates:

Overall:

MAE_all_models.png
RMSE_all_models.png
MAPE_all_models.png
R2_all_models.png


Horizon-wise:

MAE_3_min.png
MAE_5_min.png
MAE_8_min.png

RMSE_3_min.png
RMSE_5_min.png
RMSE_8_min.png

MAPE_3_min.png
MAPE_5_min.png
MAPE_8_min.png

R2_3_min.png
R2_5_min.png
R2_8_min.png

"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CSV_FILE = OUTPUT_DIR / "final_metrics.csv"


GRAPH_DIR = OUTPUT_DIR / "graphs"



GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)






# ======================================================
# Model Color Mapping
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


    if not CSV_FILE.exists():

        raise FileNotFoundError(

            f"Missing file:\n{CSV_FILE}"

        )


    df = pd.read_csv(

        CSV_FILE

    )


    return df







# ======================================================
# Overall Metric Graph
# ======================================================


def create_metric_graph(

        df,

        metric

):


    plt.figure(

        figsize=(10,6)

    )




    for model in df["Model"].unique():


        model_data = df[

            df["Model"]

            ==

            model

        ]



        plt.plot(

            model_data["Horizon"],

            model_data[metric],

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

        True,

        alpha=0.3

    )



    plt.tight_layout()




    plt.savefig(

        GRAPH_DIR /

        f"{metric}_all_models.png",

        dpi=300,

        bbox_inches="tight"

    )



    plt.close()







# ======================================================
# Horizon Bar Graph
# ======================================================


def create_horizon_graph(

        df,

        horizon

):


    data = df[

        df["Horizon"]

        ==

        horizon

    ]




    metrics = [

        "MAE",

        "RMSE",

        "MAPE",

        "R2"

    ]






    for metric in metrics:



        plt.figure(

            figsize=(8,5)

        )




        colors = [

            MODEL_COLORS.get(

                model,

                "black"

            )

            for model in data["Model"]

        ]






        plt.bar(

            data["Model"],

            data[metric],

            color=colors

        )





        plt.xlabel(

            "Model"

        )



        plt.ylabel(

            metric

        )



        plt.title(

            f"{metric} Comparison ({horizon})"

        )



        plt.xticks(

            rotation=45

        )



        plt.grid(

            axis="y",

            alpha=0.3

        )



        plt.tight_layout()




        filename = (

            f"{metric}_{horizon.replace(' ','_')}.png"

        )



        plt.savefig(

            GRAPH_DIR /

            filename,

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

        "Final Comparison Graph Generation"

    )

    print("="*70)





    df = load_data()






    metrics = [

        "MAE",

        "RMSE",

        "MAPE",

        "R2"

    ]






    for metric in metrics:


        print(

            f"Generating {metric} graph..."

        )


        create_metric_graph(

            df,

            metric

        )







    for horizon in df["Horizon"].unique():


        print(

            f"Generating {horizon} graphs..."

        )


        create_horizon_graph(

            df,

            horizon

        )







    print()

    print(

        "✓ Final graphs generated successfully"

    )


    print()

    print(

        "Saved at:"

    )


    print(

        GRAPH_DIR

    )


    print("="*70)







if __name__ == "__main__":


    main()