"""
Final Model Comparison

Models:

LSTM
GRU
GraphWaveNet
Transformer


Metrics:

MAE
RMSE
MAPE
R2


Generates:

final_metrics.csv

"""


from pathlib import Path
import json
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


OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)






MODEL_PATHS = {


"LSTM":

PROJECT_ROOT /
"comparison_models" /
"lstm" /
"outputs" /
"metrics",



"GRU":

PROJECT_ROOT /
"comparison_models" /
"gru" /
"outputs" /
"metrics",



"GraphWaveNet":

PROJECT_ROOT /
"comparison_models" /
"graph_wavenet" /
"outputs" /
"metrics",



"Transformer":

PROJECT_ROOT /
"comparison_models" /
"transformer" /
"outputs" /
"metrics"

}






HORIZONS=[3,5,8]






# ======================================================
# Collect Metrics
# ======================================================


def collect_metrics():


    results=[]




    for model,folder in MODEL_PATHS.items():



        for horizon in HORIZONS:



            file=(

                folder /

                f"{model}_{horizon}min_metrics.json"

            )



            if not file.exists():


                print(

                    "Missing:",

                    file

                )

                continue





            with open(file,"r") as f:


                metrics=json.load(f)






            results.append({


                "Model":

                model,



                "Horizon":

                f"{horizon} min",



                "MAE":

                metrics["MAE"],



                "RMSE":

                metrics["RMSE"],



                "MAPE":

                metrics["MAPE"],



                "R2":

                metrics["R2"]

            })





    return pd.DataFrame(results)







# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Final Model Comparison"

    )

    print("="*70)




    df=collect_metrics()



    output_file=(

        OUTPUT_DIR /

        "final_metrics.csv"

    )



    df.to_csv(

        output_file,

        index=False

    )





    print()

    print(

        "✓ Final Metrics Saved"

    )


    print(output_file)



    print()

    print(df)





if __name__=="__main__":


    main()