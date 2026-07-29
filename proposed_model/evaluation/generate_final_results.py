"""
Final Research Results Generator

Generates:
1. Model comparison CSV
2. Model comparison Excel
3. Horizon comparison graphs
4. Accuracy comparison
5. Error comparison
6. Loss curve comparison
7. Traffic impact analysis table

"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt




# =====================================================
# Paths
# =====================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


OUTPUT_DIR = (

    PROJECT_ROOT

    /

    "proposed_model"

    /

    "outputs"

    /

    "comparison"

)



FINAL_DIR = (

    PROJECT_ROOT

    /

    "proposed_model"

    /

    "outputs"

    /

    "final_results"

)


FINAL_DIR.mkdir(

    parents=True,

    exist_ok=True

)





# =====================================================
# Read Metrics
# =====================================================


def load_metrics():


    all_results = []



    horizons = [

        3,

        5,

        10

    ]



    for h in horizons:


        file = (

            OUTPUT_DIR

            /

            f"horizon_{h}min"

            /

            "metrics.csv"

        )



        print(

            "Reading:",

            file

        )



        df = pd.read_csv(

            file

        )



        df["Horizon"] = f"{h} min"



        all_results.append(

            df

        )



    final_df = pd.concat(

        all_results,

        ignore_index=True

    )


    return final_df







# =====================================================
# Save CSV + Excel
# =====================================================


def save_table(df):


    csv_file = (

        FINAL_DIR

        /

        "model_comparison.csv"

    )



    excel_file = (

        FINAL_DIR

        /

        "model_comparison.xlsx"

    )



    df.to_csv(

        csv_file,

        index=False

    )


    df.to_excel(

        excel_file,

        index=False

    )



    print(

        "Saved:",

        csv_file

    )


    print(

        "Saved:",

        excel_file

    )








# =====================================================
# Horizon RMSE Graph
# =====================================================


def horizon_graph(df):


    plt.figure(

        figsize=(9,6)

    )


    for model in df["Model"].unique():


        temp = df[

            df["Model"] == model

        ]



        plt.plot(

            temp["Horizon"],

            temp["RMSE"],

            marker="o",

            label=model

        )



    plt.title(

        "RMSE Variation Across Forecast Horizons"

    )


    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        "RMSE"

    )


    plt.grid(

        alpha=0.3

    )


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        FINAL_DIR

        /

        "horizon_rmse_comparison.png",

        dpi=300

    )


    plt.close()







# =====================================================
# Accuracy Graph
# =====================================================


def accuracy_graph(df):


    plt.figure(

        figsize=(9,6)

    )


    for model in df["Model"].unique():


        temp = df[

            df["Model"] == model

        ]


        plt.plot(

            temp["Horizon"],

            temp["R2"],

            marker="o",

            label=model

        )



    plt.title(

        "R2 Accuracy Comparison"

    )


    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        "R2 Score"

    )


    plt.grid(

        alpha=0.3

    )


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        FINAL_DIR

        /

        "accuracy_comparison.png",

        dpi=300

    )


    plt.close()








# =====================================================
# Error Comparison
# =====================================================


def error_graph(df):


    metrics = [

        "MSE",

        "MAE",

        "RMSE"

    ]



    for metric in metrics:


        plt.figure(

            figsize=(9,6)

        )


        avg = (

            df

            .groupby("Model")[metric]

            .mean()

        )


        avg.plot(

            kind="bar"

        )


        plt.title(

            f"{metric} Comparison"

        )


        plt.ylabel(

            metric

        )


        plt.grid(

            axis="y",

            alpha=0.3

        )


        plt.tight_layout()



        plt.savefig(

            FINAL_DIR

            /

            f"{metric.lower()}_comparison.png",

            dpi=300

        )


        plt.close()







# =====================================================
# Loss Curve Combine
# =====================================================


def loss_curve():


    loss_dir = (

        OUTPUT_DIR

        /

        "loss_curves"

    )



    files = {

        "TransGTR":

        "transgtr_loss.png",


        "LSTM":

        "lstm_loss.png",


        "GRU":

        "gru_loss.png",


        "GraphWaveNet":

        "gwnet_loss.png"

    }



    print(

        "Loss curves found"

    )



    print(

        files

    )







# =====================================================
# Traffic Impact Table
# =====================================================


def traffic_analysis():


    traffic_dir = (

        OUTPUT_DIR

        /

        "traffic_metrics"

    )



    output = []



    for file in [

        "queue_length.csv",

        "waiting_time.csv",

        "spillback.csv"

    ]:


        path = traffic_dir / file



        if path.exists():


            df = pd.read_csv(path)


            df["Metric"] = file.replace(

                ".csv",

                ""

            )


            output.append(df)





    if output:


        result = pd.concat(

            output,

            ignore_index=True

        )



        result.to_csv(

            FINAL_DIR

            /

            "traffic_impact_analysis.csv",

            index=False

        )



        print(

            "Traffic analysis saved"

        )








# =====================================================
# MAIN
# =====================================================


def main():


    print("="*70)

    print(

        "Generating Final Research Results"

    )

    print("="*70)



    df = load_metrics()



    save_table(

        df

    )


    horizon_graph(

        df

    )


    accuracy_graph(

        df

    )


    error_graph(

        df

    )


    loss_curve()



    traffic_analysis()



    print("="*70)

    print(

        "FINAL RESULTS GENERATED"

    )

    print("="*70)






if __name__ == "__main__":

    main()