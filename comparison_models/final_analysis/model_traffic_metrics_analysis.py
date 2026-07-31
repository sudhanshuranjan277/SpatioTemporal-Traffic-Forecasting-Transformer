"""
Model Traffic Metrics Analysis

Generates model-wise:

- Waiting Time
- Queue Length
- Spillback Events

for:

LSTM
GRU
GraphWaveNet
Transformer


Uses prediction outputs (.pt)

Output:
model_traffic_impact_results.csv
"""


from pathlib import Path
import torch
import pandas as pd
import numpy as np



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



RESULT_FILE = (
    OUTPUT_DIR
    /
    "model_traffic_impact_results.csv"
)





# ======================================================
# Prediction Paths
# ======================================================


PREDICTION_PATHS = {


"LSTM":

PROJECT_ROOT
/
"comparison_models"
/
"lstm"
/
"outputs"
/
"predictions",



"GRU":

PROJECT_ROOT
/
"comparison_models"
/
"gru"
/
"outputs"
/
"predictions",



"GraphWaveNet":

PROJECT_ROOT
/
"comparison_models"
/
"graph_wavenet"
/
"outputs"
/
"predictions",



"Transformer":

PROJECT_ROOT
/
"comparison_models"
/
"transformer"
/
"outputs"
/
"predictions"

}




HORIZONS = [3,5,8]







# ======================================================
# Impact Calculation
# ======================================================


def calculate_metrics(prediction):


    """
    Convert traffic flow prediction
    into traffic impact indicators.
    """


    values = prediction.numpy()



    mean_flow = np.mean(values)



    max_flow = np.max(values)



    std_flow = np.std(values)




    # Estimated metrics

    waiting_time = (
        mean_flow * 0.85
    )



    queue_length = (
        max_flow * 0.25
    )



    spillback_events = int(
        std_flow
        >
        5
    )



    return (


        round(waiting_time,4),


        round(queue_length,4),


        spillback_events


    )







# ======================================================
# Process Models
# ======================================================


def process_predictions():



    results = []



    for model,folder in PREDICTION_PATHS.items():



        for horizon in HORIZONS:



            file = (

                folder

                /

                f"{model}_{horizon}min_predictions.pt"

            )



            if not file.exists():


                print(

                    "Missing:",

                    file

                )

                continue





            data = torch.load(

                file,

                map_location="cpu"

            )



            prediction = data["prediction"]




            waiting_time, queue_length, spillback = calculate_metrics(

                prediction

            )





            results.append({


                "Model":

                model,



                "Horizon":

                f"{horizon} min",



                "Waiting_Time":

                waiting_time,



                "Queue_Length":

                queue_length,



                "Spillback_Events":

                spillback



            })



    return pd.DataFrame(results)









# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Model Traffic Impact Analysis"

    )

    print("="*70)





    df = process_predictions()





    df.to_csv(

        RESULT_FILE,

        index=False

    )





    print()

    print(

        "✓ Generated Successfully"

    )


    print(

        RESULT_FILE

    )


    print()

    print(df)






if __name__=="__main__":

    main()