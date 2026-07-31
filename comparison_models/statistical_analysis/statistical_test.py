"""
Statistical Analysis Graph Generator

Supports:

LSTM
GRU
GraphWaveNet
Transformer


Reads:

statistical_results.csv


Generates:

- Mean Error Comparison
- Standard Deviation Comparison
- Confidence Interval Comparison
- Statistical Test p-values

"""


from pathlib import Path
import re


import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CSV_FILE = OUTPUT_DIR / "statistical_results.csv"


GRAPH_DIR = OUTPUT_DIR / "graphs"



GRAPH_DIR.mkdir(

    parents=True,

    exist_ok=True

)






# ======================================================
# Load CSV
# ======================================================


def load_data():


    df = pd.read_csv(

        CSV_FILE

    )


    return df







# ======================================================
# Mean Error Graph
# ======================================================


def mean_error_graph(df):


    plt.figure(

        figsize=(12,6)

    )



    x = df["Comparison"]



    model1_values = []


    model2_values = []



    labels1=[]

    labels2=[]



    for _,row in df.iterrows():


        comparison = row["Comparison"]


        m1,m2 = comparison.split(" vs ")



        labels1.append(m1)

        labels2.append(m2)



        model1_values.append(

            row[f"{m1} Mean Error"]

        )


        model2_values.append(

            row[f"{m2} Mean Error"]

        )





    plt.plot(

        x,

        model1_values,

        marker="o",

        label="Model 1"

    )



    plt.plot(

        x,

        model2_values,

        marker="o",

        label="Model 2"

    )



    plt.xticks(

        rotation=60,

        ha="right"

    )


    plt.ylabel(

        "Mean Absolute Error"

    )


    plt.title(

        "Mean Error Comparison"

    )


    plt.legend()

    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        "mean_error_comparison.png",

        dpi=300

    )



    plt.close()







# ======================================================
# Standard Deviation
# ======================================================


def std_graph(df):


    plt.figure(

        figsize=(12,6)

    )


    model1=[]

    model2=[]


    labels=[]



    for _,row in df.iterrows():


        m1,m2=row["Comparison"].split(" vs ")


        labels.append(

            row["Comparison"]

        )


        model1.append(

            row[f"{m1} Std"]

        )


        model2.append(

            row[f"{m2} Std"]

        )






    plt.plot(

        labels,

        model1,

        marker="o",

        label="Model 1"

    )


    plt.plot(

        labels,

        model2,

        marker="o",

        label="Model 2"

    )



    plt.xticks(

        rotation=60,

        ha="right"

    )


    plt.ylabel(

        "Standard Deviation"

    )


    plt.title(

        "Standard Deviation Comparison"

    )


    plt.legend()

    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        "standard_deviation_comparison.png",

        dpi=300

    )



    plt.close()







# ======================================================
# Confidence Interval
# ======================================================


def confidence_graph(df):


    plt.figure(

        figsize=(12,6)

    )


    labels=[]


    values=[]



    for _,row in df.iterrows():


        labels.append(

            row["Comparison"]

        )



        ci = row["95% CI Difference"]




        nums = re.findall(

            r"[-+]?\d*\.\d+",

            ci

        )



        if len(nums)==2:


            low=float(nums[0])


            high=float(nums[1])



            values.append(

                (

                    high-low

                )

            )


        else:

            values.append(0)






    plt.bar(

        labels,

        values

    )



    plt.xticks(

        rotation=60,

        ha="right"

    )


    plt.ylabel(

        "Confidence Interval Width"

    )


    plt.title(

        "95% Confidence Interval Comparison"

    )


    plt.grid(

        axis="y"

    )



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        "confidence_interval_comparison.png",

        dpi=300

    )



    plt.close()








# ======================================================
# P-values
# ======================================================


def pvalue_graph(df):


    plt.figure(

        figsize=(12,6)

    )



    labels=df["Comparison"]



    t_values=df["T-test p-value"]


    w_values=df["Wilcoxon p-value"]






    plt.plot(

        labels,

        t_values,

        marker="o",

        label="T-test"

    )



    plt.plot(

        labels,

        w_values,

        marker="o",

        label="Wilcoxon"

    )




    plt.xticks(

        rotation=60,

        ha="right"

    )



    plt.ylabel(

        "p-value"

    )



    plt.yscale(

        "log"

    )


    plt.title(

        "Statistical Test p-values"

    )


    plt.legend()


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        "statistical_test_pvalues.png",

        dpi=300

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



    mean_error_graph(df)



    std_graph(df)



    confidence_graph(df)



    pvalue_graph(df)






    print()

    print(

        "✓ Statistical graphs generated"

    )



    print()

    print(

        GRAPH_DIR

    )


    print("="*70)







if __name__=="__main__":


    main()