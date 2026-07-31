"""
Final Result Summary Generator

Reads:

final_metrics.csv


Generates:

1. best_model_summary.csv
2. improvement_analysis.csv


Models:

LSTM
GRU
GraphWaveNet
Transformer

"""


from pathlib import Path

import pandas as pd





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CSV_FILE = OUTPUT_DIR / "final_metrics.csv"





# ======================================================
# Load Data
# ======================================================


def load_data():


    if not CSV_FILE.exists():

        raise FileNotFoundError(

            CSV_FILE

        )


    return pd.read_csv(

        CSV_FILE

    )






# ======================================================
# Best Model
# ======================================================


def best_model_summary(df):


    results=[]



    for horizon in df["Horizon"].unique():


        data = df[

            df["Horizon"] == horizon

        ]



        # lowest MAE


        best = data.loc[

            data["MAE"].idxmin()

        ]




        results.append({


            "Horizon":

            horizon,


            "Best Model":

            best["Model"],


            "MAE":

            best["MAE"],


            "RMSE":

            best["RMSE"],


            "MAPE":

            best["MAPE"],


            "R2":

            best["R2"]


        })



    return pd.DataFrame(results)







# ======================================================
# Improvement Analysis
# ======================================================


def improvement_analysis(df):


    results=[]




    for horizon in df["Horizon"].unique():



        data = df[

            df["Horizon"] == horizon

        ]



        baseline = data[

            data["Model"]

            ==

            "LSTM"

        ].iloc[0]




        best = data.loc[

            data["MAE"].idxmin()

        ]





        improvement = (

            (

                baseline["MAE"]

                -

                best["MAE"]

            )

            /

            baseline["MAE"]

        ) * 100





        results.append({



            "Horizon":

            horizon,



            "Baseline Model":

            "LSTM",



            "Best Model":

            best["Model"],



            "Baseline MAE":

            baseline["MAE"],



            "Best MAE":

            best["MAE"],



            "MAE Improvement %":

            round(

                improvement,

                3

            )

        })




    return pd.DataFrame(results)







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Final Result Summary Generation"

    )

    print("="*70)





    df = load_data()





    best_df = best_model_summary(

        df

    )



    improvement_df = improvement_analysis(

        df

    )





    best_file = (

        OUTPUT_DIR /

        "best_model_summary.csv"

    )



    improvement_file = (

        OUTPUT_DIR /

        "improvement_analysis.csv"

    )





    best_df.to_csv(

        best_file,

        index=False

    )



    improvement_df.to_csv(

        improvement_file,

        index=False

    )





    print()

    print(

        "✓ Best model summary saved"

    )


    print(best_file)



    print()

    print(

        "✓ Improvement analysis saved"

    )


    print(improvement_file)



    print()

    print(best_df)

    print()

    print(improvement_df)






if __name__=="__main__":

    main()