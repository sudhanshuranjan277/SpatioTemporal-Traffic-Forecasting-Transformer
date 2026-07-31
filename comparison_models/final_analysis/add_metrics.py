"""
Add MSE and Accuracy Metrics

Input:
final_metrics.csv

Output:
final_metrics_updated.csv


Metrics Added:

MSE
Accuracy
"""


from pathlib import Path

import pandas as pd





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


INPUT_FILE = OUTPUT_DIR / "final_metrics.csv"


OUTPUT_FILE = OUTPUT_DIR / "final_metrics_updated.csv"







# ======================================================
# Main
# ======================================================


def main():


    print("=" * 70)

    print("Updating Forecasting Metrics")

    print("=" * 70)





    # Load existing metrics

    df = pd.read_csv(

        INPUT_FILE

    )





    # ------------------------------
    # Add MSE
    # ------------------------------

    # MSE = RMSE squared

    df["MSE"] = (

        df["RMSE"] ** 2

    )






    # ------------------------------
    # Add Accuracy
    # ------------------------------

    # Regression accuracy representation

    # Accuracy = 100/(1+MAPE/100)

    df["Accuracy"] = (

        100 /

        (

            1 +

            df["MAPE"] / 100

        )

    )






    # Round values

    df["MSE"] = df["MSE"].round(4)

    df["Accuracy"] = df["Accuracy"].round(2)







    # Reorder columns


    df = df[

        [

            "Model",

            "Horizon",

            "MAE",

            "MSE",

            "RMSE",

            "MAPE",

            "R2",

            "Accuracy"

        ]

    ]






    # Save file


    df.to_csv(

        OUTPUT_FILE,

        index=False

    )






    print()

    print("✓ Updated metrics generated")

    print()

    print(

        OUTPUT_FILE

    )


    print()

    print(df)

    print()





if __name__ == "__main__":

    main()