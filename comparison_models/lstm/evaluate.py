"""
LSTM Evaluation

Dynamic Horizon Evaluation

Supports:
3 min
5 min
8 min
"""


from pathlib import Path
import json


import torch
from torch.utils.data import DataLoader


from proposed_model.configs.config import (
    DEVICE,
)


from comparison_models.lstm.model import (
    LSTMBaseline
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
    CURRENT_DIR
    /
    "outputs"
)


CHECKPOINT_DIR = (
    OUTPUT_DIR
    /
    "checkpoints"
)


METRIC_DIR = (
    OUTPUT_DIR
    /
    "metrics"
)


PREDICTION_DIR = (
    OUTPUT_DIR
    /
    "predictions"
)





# ======================================================
# Evaluation Function
# ======================================================


def evaluate(horizon):


    print("="*70)

    print(
        f"LSTM Evaluation | Horizon {horizon} min"
    )

    print("="*70)




    checkpoint_path = (

        CHECKPOINT_DIR
        /
        f"LSTM_{horizon}min.pth"

    )



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




    # -----------------------------
    # Model
    # -----------------------------


    model = LSTMBaseline(

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



    predictions = []

    targets = []




    with torch.no_grad():


        for x,y in loader:


            x = x.to(DEVICE)

            y = y.to(DEVICE)



            prediction = model(x)



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





    # -----------------------------
    # Metrics
    # -----------------------------


    results = calculate_metrics(

        predictions,

        targets

    )




    print()

    print("="*70)

    print(

        "Results"

    )

    print("="*70)



    for key,value in results.items():

        print(

            f"{key}: {value:.6f}"

        )



    print("="*70)





    # -----------------------------
    # Save
    # -----------------------------


    METRIC_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    PREDICTION_DIR.mkdir(

        parents=True,

        exist_ok=True

    )





    with open(

        METRIC_DIR
        /
        f"LSTM_{horizon}min_metrics.json",

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

        PREDICTION_DIR
        /
        f"LSTM_{horizon}min_predictions.pt"

    )





    print()

    print(

        "Metrics Saved:",

        METRIC_DIR
        /
        f"LSTM_{horizon}min_metrics.json"

    )



    print(

        "Predictions Saved:",

        PREDICTION_DIR
        /
        f"LSTM_{horizon}min_predictions.pt"

    )



    return results





# ======================================================
# Direct Run
# ======================================================


if __name__ == "__main__":


    evaluate(5)