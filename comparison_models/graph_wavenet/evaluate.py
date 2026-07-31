"""
Graph WaveNet Evaluation

Runs evaluation for:

3 min
5 min
8 min

Outputs:

outputs/
    metrics/
    predictions/
"""


from pathlib import Path
import json


import torch
from torch.utils.data import DataLoader


from proposed_model.configs.config import DEVICE


from proposed_model.data.dataset import TrafficDataset


from comparison_models.graph_wavenet.model import GraphWaveNet


from proposed_model.evaluation.metrics import calculate_metrics





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


METRIC_DIR = OUTPUT_DIR / "metrics"


PREDICTION_DIR = OUTPUT_DIR / "predictions"



HORIZONS = [3,5,8]






# ======================================================
# Evaluation
# ======================================================


def evaluate(horizon):


    print("="*70)

    print(
        f"Graph WaveNet Evaluation | Horizon {horizon} min"
    )

    print("="*70)




    checkpoint_path = (

        CHECKPOINT_DIR /

        f"GraphWaveNet_{horizon}min.pth"

    )




    # -----------------------------
    # Dataset
    # -----------------------------


    dataset = TrafficDataset(

        split="test",

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


    x_sample, _ = dataset[0]


    num_nodes = x_sample.shape[1]



    model = GraphWaveNet(

        num_nodes=num_nodes,

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



    print(
        "Checkpoint Loaded:"
    )


    print(
        checkpoint_path
    )






    predictions = []

    targets = []




    with torch.no_grad():


        for x,y in loader:


            x = x.to(DEVICE)

            y = y.to(DEVICE)



            pred = model(x)



            predictions.append(

                pred.cpu()

            )


            targets.append(

                y.cpu()

            )





    predictions = torch.cat(predictions)

    targets = torch.cat(targets)




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
        "Evaluation Results"
    )

    print("="*70)



    for k,v in results.items():

        print(
            f"{k}: {v:.6f}"
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




    metric_file = (

        METRIC_DIR /

        f"GraphWaveNet_{horizon}min_metrics.json"

    )



    with open(metric_file,"w") as f:

        json.dump(

            results,

            f,

            indent=4

        )






    prediction_file = (

        PREDICTION_DIR /

        f"GraphWaveNet_{horizon}min_predictions.pt"

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






# ======================================================
# Main
# ======================================================


def main():


    for horizon in HORIZONS:


        evaluate(horizon)





if __name__ == "__main__":

    main()