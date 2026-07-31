"""
GRU Evaluation

Evaluates GRU model for different forecast horizons:

3 minutes
5 minutes
8 minutes


Input:
Previous 12 minutes traffic observations

Output:
Future traffic flow prediction


Metrics:
MAE
RMSE
MAPE
R2
"""


from pathlib import Path
import json


import torch
from torch.utils.data import DataLoader



from proposed_model.configs.config import (
    DEVICE,
)



from comparison_models.gru.model import (
    GRUBaseline
)



from proposed_model.data.evaluation_dataset import (
    EvaluationDataset
)



from proposed_model.evaluation.metrics import (
    calculate_metrics
)





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = (
    CURRENT_DIR /
    "outputs"
)


CHECKPOINT_DIR = (
    OUTPUT_DIR /
    "checkpoints"
)


METRIC_DIR = (
    OUTPUT_DIR /
    "metrics"
)


PREDICTION_DIR = (
    OUTPUT_DIR /
    "predictions"
)





# ======================================================
# Evaluation Function
# ======================================================


def evaluate(horizon):


    print("=" * 70)

    print(
        f"GRU Evaluation | Horizon {horizon} min"
    )

    print("=" * 70)




    checkpoint_path = (

        CHECKPOINT_DIR
        /
        f"GRU_{horizon}min.pth"

    )



    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n{checkpoint_path}"

        )





    # ==================================================
    # Dataset
    # ==================================================


    dataset = EvaluationDataset(

        horizon=horizon

    )


    loader = DataLoader(

        dataset,

        batch_size=32,

        shuffle=False

    )



    print(

        "Test Samples:",

        len(dataset)

    )





    # ==================================================
    # Model
    # ==================================================


    model = GRUBaseline(

        horizon=horizon

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE

    )



    model.load_state_dict(

        checkpoint["model_state_dict"]

    )



    model.to(DEVICE)


    model.eval()



    print()

    print(
        "Checkpoint Loaded:"
    )


    print(

        checkpoint_path

    )






    # ==================================================
    # Prediction
    # ==================================================


    predictions = []

    targets = []





    with torch.no_grad():


        for x,y in loader:


            x = x.to(DEVICE)

            y = y.to(DEVICE)



            prediction = model(

                x

            )



            predictions.append(

                prediction.cpu()

            )


            targets.append(

                y.cpu()

            )





    predictions = torch.cat(

        predictions

    )


    targets = torch.cat(

        targets

    )





    print()

    print(

        "Prediction Shape:",

        predictions.shape

    )


    print(

        "Target Shape:",

        targets.shape

    )





    # ==================================================
    # Metrics
    # ==================================================


    results = calculate_metrics(

        predictions,

        targets

    )





    print()

    print("=" * 70)

    print(

        "Evaluation Results"

    )

    print("=" * 70)



    for key,value in results.items():


        print(

            f"{key}: {value:.6f}"

        )



    print("=" * 70)






    # ==================================================
    # Save Results
    # ==================================================


    METRIC_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    PREDICTION_DIR.mkdir(

        parents=True,

        exist_ok=True

    )





    metric_file = (

        METRIC_DIR
        /
        f"GRU_{horizon}min_metrics.json"

    )



    prediction_file = (

        PREDICTION_DIR
        /
        f"GRU_{horizon}min_predictions.pt"

    )





    with open(

        metric_file,

        "w"

    ) as f:


        json.dump(

            results,

            f,

            indent=4

        )





    torch.save(

        {

            "prediction": predictions,

            "target": targets

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



    print("=" * 70)



    return results





# ======================================================
# Direct Run
# ======================================================


if __name__ == "__main__":


    evaluate(5)