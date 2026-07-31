"""
Final Combined Traffic Forecasting Analysis

Combines:

1. Forecasting Metrics
   - MAE
   - MSE
   - RMSE
   - Accuracy

2. Traffic Impact Metrics
   - Waiting Time
   - Queue Length
   - Spillback Events


Output:

final_combined_results.csv

"""


from pathlib import Path
import pandas as pd





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



FORECAST_FILE = (

    OUTPUT_DIR

    /

    "final_metrics.csv"

)



TRAFFIC_FILE = (

    OUTPUT_DIR

    /

    "traffic_impact_results.csv"

)



OUTPUT_FILE = (

    OUTPUT_DIR

    /

    "final_combined_results.csv"

)







# ======================================================
# Load Data
# ======================================================


def load_data():


    forecast = pd.read_csv(

        FORECAST_FILE

    )


    traffic = pd.read_csv(

        TRAFFIC_FILE

    )


    return forecast, traffic







# ======================================================
# Combine Analysis
# ======================================================


def combine_results():



    forecast, traffic = load_data()





    # Average traffic impact across junctions

    traffic_summary = {


        "Waiting_Time":

        traffic["Average Waiting Time"].mean(),



        "Queue_Length":

        traffic["Average Queue Length"].mean(),



        "Spillback_Events":

        traffic["Spillback Events"].sum()


    }







    # Add same traffic impact
    # for all forecasting models


    forecast["Waiting_Time"] = (

        traffic_summary["Waiting_Time"]

    )



    forecast["Queue_Length"] = (

        traffic_summary["Queue_Length"]

    )



    forecast["Spillback_Events"] = (

        traffic_summary["Spillback_Events"]

    )








    return forecast







# ======================================================
# Save
# ======================================================


def main():


    print("="*70)

    print(

        "Final Combined Analysis"

    )

    print("="*70)





    df = combine_results()





    columns = [

        "Model",

        "Horizon",

        "MAE",

        "MSE",

        "RMSE",

        "MAPE",

        "R2",

        "Accuracy",

        "Waiting_Time",

        "Queue_Length",

        "Spillback_Events"

    ]





    df = df[columns]





    df = df.round(4)





    df.to_csv(

        OUTPUT_FILE,

        index=False

    )






    print()

    print(

        "✓ Final Combined Results Generated"

    )



    print()

    print(

        OUTPUT_FILE

    )


    print()

    print(df)







if __name__=="__main__":

    main()