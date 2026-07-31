"""
Transformer Evaluation Module

Calculates:

MAE
RMSE
MAPE
R2 Score


Generates:

Prediction files
Metric JSON files


Supports:

3 min
5 min
8 min

"""


from pathlib import Path
import json


import torch
import numpy as np


from torch.utils.data import DataLoader


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


from comparison_models.transformer.model import TrafficTransformer


from proposed_model.data.dataset import TrafficDataset





# ======================================================
# Device
# ======================================================


DEVICE = (

    torch.device("cuda")

    if torch.cuda.is_available()

    else torch.device("cpu")

)





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


METRIC_DIR = OUTPUT_DIR / "metrics"


PREDICTION_DIR = OUTPUT_DIR / "predictions"



METRIC_DIR.mkdir(

    parents=True,

    exist_ok=True

)


PREDICTION_DIR.mkdir(

    parents=True,

    exist_ok=True

)







# ======================================================
# Metrics
# ======================================================


def calculate_metrics(

        y_true,

        y_pred

):


    y_true = y_true.flatten()

    y_pred = y_pred.flatten()




    mae = mean_absolute_error(

        y_true,

        y_pred

    )



    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )




    # MAPE

    epsilon = 1e-8


    mape = np.mean(

        np.abs(

            (

                y_true - y_pred

            )

            /

            (

                y_true + epsilon

            )

        )

    ) * 100





    r2 = r2_score(

        y_true,

        y_pred

    )



    return {


        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape),

        "R2": float(r2)

    }









# ======================================================
# Evaluation
# ======================================================


def evaluate(horizon):


    print("="*70)

    print(

        f"Transformer Evaluation | Horizon {horizon} min"

    )

    print("="*70)





    checkpoint = (

        CHECKPOINT_DIR

        /

        f"Transformer_{horizon}min.pth"

    )



    if not checkpoint.exists():

        raise FileNotFoundError(

            f"Checkpoint not found: {checkpoint}"

        )






    # Dataset


    test_dataset = TrafficDataset(

        split="test",

        horizon=horizon

    )




    test_loader = DataLoader(

        test_dataset,

        batch_size=16,

        shuffle=False

    )






    # Model


    model = TrafficTransformer(

        horizon=horizon

    )



    checkpoint_data = torch.load(

        checkpoint,

        map_location=DEVICE

    )



    model.load_state_dict(

        checkpoint_data["model_state_dict"]

    )



    model.to(DEVICE)


    model.eval()





    predictions = []

    targets = []





    with torch.no_grad():



        for x,y in test_loader:



            x = x.to(DEVICE)

            y = y.to(DEVICE)




            output = model(x)




            predictions.append(

                output.cpu()

            )


            targets.append(

                y.cpu()

            )






    predictions = torch.cat(

        predictions,

        dim=0

    )



    targets = torch.cat(

        targets,

        dim=0

    )






    # Metrics


    metrics = calculate_metrics(

        targets.numpy(),

        predictions.numpy()

    )







    print()

    print("="*70)

    print(

        "Evaluation Results"

    )

    print("="*70)



    for key,value in metrics.items():


        print(

            f"{key}: {value:.6f}"

        )



    print("="*70)








    # Save metrics


    metric_file = (

        METRIC_DIR

        /

        f"Transformer_{horizon}min_metrics.json"

    )



    with open(

        metric_file,

        "w"

    ) as f:


        json.dump(

            metrics,

            f,

            indent=4

        )







    # Save predictions


    prediction_file = (

        PREDICTION_DIR

        /

        f"Transformer_{horizon}min_predictions.pt"

    )



    torch.save(

        {

            "prediction":

                predictions,

            "target":

                targets

        },

        prediction_file

    )






    print()

    print(

        "Saved Metrics:"

    )

    print(

        metric_file

    )


    print()

    print(

        "Saved Predictions:"

    )

    print(

        prediction_file

    )



    print("="*70)








# ======================================================
# Main
# ======================================================


if __name__ == "__main__":


    evaluate(3)
    
    evaluate(5)
    
    evaluate(8)