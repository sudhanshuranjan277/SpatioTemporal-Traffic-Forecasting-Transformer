"""
Statistical Analysis Graph Generator


Supports:

LSTM
GRU
Graph WaveNet


Reads:

outputs/statistical_results.csv


Generates:

1. Mean Error Comparison
2. Standard Deviation Comparison
3. Confidence Interval Comparison
4. Statistical Test p-value Comparison

"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CSV_FILE = (

    OUTPUT_DIR /

    "statistical_results.csv"

)



GRAPH_DIR = (

    OUTPUT_DIR /

    "graphs"

)


GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)







# ======================================================
# Load CSV
# ======================================================


def load_data():


    if not CSV_FILE.exists():

        raise FileNotFoundError(

            f"File not found: {CSV_FILE}"

        )


    return pd.read_csv(

        CSV_FILE

    )








# ======================================================
# Generic Metric Plot
# ======================================================


def plot_metric(

        df,

        columns,

        ylabel,

        title,

        filename

):


    plt.figure(

        figsize=(10,6)

    )



    for column,label in columns:


        if column in df.columns:


            plt.plot(

                df["Horizon"],

                df[column],

                marker="o",

                label=label

            )



    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        ylabel

    )


    plt.title(

        title

    )


    plt.legend()


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR / filename,

        dpi=300,

        bbox_inches="tight"

    )



    plt.close()







# ======================================================
# Confidence Interval Parser
# ======================================================


def parse_ci(value):


    value = str(value)



    value = value.replace(

        "np.float64",

        ""

    )


    value = value.replace(

        "(",

        ""

    )


    value = value.replace(

        ")",

        ""

    )


    value = value.replace(

        " ",

        ""

    )



    low, high = value.split(",")



    return (

        float(low),

        float(high)

    )









# ======================================================
# Confidence Interval Graph
# ======================================================


def confidence_interval_graph(df):


    models = []



    for model in [

        "LSTM",

        "GRU",

        "GraphWaveNet"

    ]:


        if f"{model} 95% CI" in df.columns:


            models.append(model)






    if len(models) == 0:


        print(

            "No Confidence Interval columns found"

        )

        return







    plt.figure(

        figsize=(10,6)

    )




    for model in models:


        values=[]



        for item in df[f"{model} 95% CI"]:


            low, high = parse_ci(item)



            values.append(

                (low + high) / 2

            )




        plt.plot(

            df["Horizon"],

            values,

            marker="o",

            label=model

        )






    plt.xlabel(

        "Forecast Horizon"

    )


    plt.ylabel(

        "Confidence Interval Mean"

    )


    plt.title(

        "95% Confidence Interval Comparison"

    )


    plt.legend()


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        "confidence_interval_comparison.png",

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

        "Statistical Graph Generation"

    )

    print("="*70)




    df = load_data()




    print()

    print(

        "Available Columns:"

    )


    print(

        list(df.columns)

    )







    # Mean Error


    plot_metric(

        df,


        [

            (

            "LSTM Mean Error",

            "LSTM"

            ),


            (

            "GRU Mean Error",

            "GRU"

            ),


            (

            "GraphWaveNet Mean Error",

            "Graph WaveNet"

            )

        ],


        "Mean Error",


        "Mean Error Comparison",


        "mean_error_comparison.png"

    )








    # Standard Deviation


    plot_metric(

        df,


        [

            (

            "LSTM Std",

            "LSTM"

            ),


            (

            "GRU Std",

            "GRU"

            ),


            (

            "GraphWaveNet Std",

            "Graph WaveNet"

            )

        ],


        "Standard Deviation",


        "Prediction Error Standard Deviation",


        "standard_deviation_comparison.png"

    )







    # Confidence Interval


    confidence_interval_graph(df)







    # Statistical Tests


    plot_metric(

        df,


        [

            (

            "T-test p-value",

            "T-test"

            ),


            (

            "Wilcoxon p-value",

            "Wilcoxon"

            )

        ],


        "p-value",


        "Statistical Test p-value Comparison",


        "statistical_test_pvalues.png"

    )







    print()

    print(

        "✓ Statistical graphs generated"

    )


    print()

    print(

        "Saved Location:"

    )


    print(

        GRAPH_DIR

    )


    print("="*70)







if __name__ == "__main__":


    main()