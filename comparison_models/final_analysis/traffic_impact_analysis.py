"""
Traffic Impact Analysis

Generates:

1. Waiting Time Comparison
2. Queue Length Comparison
3. Spillback Event Comparison

Output:

final_analysis/outputs/traffic_impact_results.csv

"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt





# ======================================================
# Paths
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRAFFIC_DIR = (

    PROJECT_ROOT
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
    /
    "traffic_metrics"

)



OUTPUT_DIR = (

    PROJECT_ROOT
    /
    "comparison_models"
    /
    "final_analysis"
    /
    "outputs"

)


GRAPH_DIR = OUTPUT_DIR / "traffic_graphs"


GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)






# ======================================================
# Load Data
# ======================================================


def load_data():


    waiting = pd.read_csv(

        TRAFFIC_DIR /
        "waiting_time.csv"

    )


    queue = pd.read_csv(

        TRAFFIC_DIR /
        "queue_length.csv"

    )


    spill = pd.read_csv(

        TRAFFIC_DIR /
        "spillback.csv"

    )


    return waiting, queue, spill







# ======================================================
# Analysis
# ======================================================


def generate_results():


    waiting, queue, spill = load_data()



    results = []



    junctions = waiting["junction_id"].unique()



    for junction in junctions:



        w = waiting[

            waiting["junction_id"]

            ==

            junction

        ]



        q = queue[

            queue["junction_id"]

            ==

            junction

        ]



        s = spill[

            spill["junction_id"]

            ==

            junction

        ]




        results.append({


            "Junction":

            junction,


            "Average Waiting Time":

            w["waiting_time"].mean(),



            "Average Queue Length":

            q["queue_length"].mean(),



            "Average Downstream Queue":

            q["downstream_queue_length"].mean(),



            "Spillback Events":

            s["spillback"]

            .astype(bool)

            .sum()



        })






    df = pd.DataFrame(results)



    df.to_csv(

        OUTPUT_DIR /

        "traffic_impact_results.csv",

        index=False

    )


    return df







# ======================================================
# Graphs
# ======================================================


def create_graph(df,column,title,filename):


    plt.figure(

        figsize=(8,5)

    )


    plt.bar(

        df["Junction"],

        df[column]

    )


    plt.xlabel(

        "Junction"

    )


    plt.ylabel(

        column

    )


    plt.title(

        title

    )


    plt.grid(

        alpha=0.3

    )


    plt.tight_layout()



    plt.savefig(

        GRAPH_DIR /

        filename,

        dpi=300

    )



    plt.close()







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Traffic Impact Analysis"

    )

    print("="*70)



    df = generate_results()



    create_graph(

        df,

        "Average Waiting Time",

        "Average Waiting Time Comparison",

        "waiting_time_comparison.png"

    )



    create_graph(

        df,

        "Average Queue Length",

        "Average Queue Length Comparison",

        "queue_length_comparison.png"

    )



    create_graph(

        df,

        "Spillback Events",

        "Spillback Event Comparison",

        "spillback_comparison.png"

    )



    print()

    print(

        "✓ Traffic Impact Analysis Completed"

    )


    print()

    print(df)






if __name__=="__main__":

    main()