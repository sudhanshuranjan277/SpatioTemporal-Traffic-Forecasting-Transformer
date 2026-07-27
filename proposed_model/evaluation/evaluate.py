"""
evaluation/evaluate.py

Model Evaluation Pipeline

Flow
----

Best Checkpoint
        |
        ↓
Load TransGTR
        |
        ↓
Test Dataset
        |
        ↓
Prediction
        |
        ↓
Metrics
        |
        ↓
Save Results
"""


from __future__ import annotations


import json

from pathlib import Path


import torch

from torch.utils.data import DataLoader



from configs.config import (
    BATCH_SIZE,
    DEVICE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


from data.dataset import TrafficDataset


from models.transgtr import TransGTR


from evaluation.metrics import (
    evaluate_batch,
    average_metrics,
)





# ==========================================================
# Device
# ==========================================================


device = DEVICE





# ==========================================================
# Load Test Dataset
# ==========================================================


def create_test_loader():

    test_dataset = TrafficDataset(
        split="test"
    )


    test_loader = DataLoader(

        dataset=test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        num_workers=0,

    )


    return test_loader

# ==========================================================
# Model Loading
# ==========================================================


def load_model():

    model = TransGTR()

    model = model.to(
        device
    )


    checkpoint_path = (
        CHECKPOINT_DIR
        /
        "best_model.pth"
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    print("=" * 70)

    print(
        "Model Loaded Successfully"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    print(
        f"Epoch      : {checkpoint['epoch']}"
    )

    print(
        f"Loss       : {checkpoint['loss']}"
    )

    print("=" * 70)


    return model





# ==========================================================
# Evaluation
# ==========================================================


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
):

    metric_results = []


    all_predictions = []

    all_targets = []



    for batch in test_loader:


        x, y = batch


        x = x.to(
            device
        )


        y = y.to(
            device
        )



        prediction = model(x)



        batch_metrics = evaluate_batch(

            prediction,

            y,

        )


        metric_results.append(
            batch_metrics
        )


        all_predictions.append(
            prediction.cpu()
        )


        all_targets.append(
            y.cpu()
        )



    final_metrics = average_metrics(
        metric_results
    )


    predictions = torch.cat(
        all_predictions,
        dim=0,
    )


    targets = torch.cat(
        all_targets,
        dim=0,
    )



    return (
        final_metrics,
        predictions,
        targets,
    )
    
    # ==========================================================
# Save Results
# ==========================================================


def save_results(
    metrics,
    predictions,
    targets,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------
    # Save Metrics
    # -------------------------------

    metrics_path = (
        OUTPUT_DIR
        /
        "test_metrics.json"
    )


    with open(
        metrics_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


    # -------------------------------
    # Save Predictions
    # -------------------------------

    prediction_path = (
        OUTPUT_DIR
        /
        "test_predictions.pt"
    )


    torch.save(

        {
            "predictions":
                predictions,

            "targets":
                targets,

        },

        prediction_path,

    )


    print("=" * 70)

    print(
        "Evaluation Results Saved"
    )

    print(
        f"Metrics      : {metrics_path}"
    )

    print(
        f"Predictions  : {prediction_path}"
    )

    print("=" * 70)





# ==========================================================
# Main
# ==========================================================


def main():


    print("=" * 70)

    print(
        "TransGTR Evaluation"
    )

    print("=" * 70)



    test_loader = create_test_loader()



    model = load_model()



    metrics, predictions, targets = evaluate_model(

        model,

        test_loader,

    )



    print()

    print("=" * 70)

    print(
        "Test Metrics"
    )

    print("=" * 70)



    for name, value in metrics.items():

        print(
            f"{name}: {value:.6f}"
        )



    print("=" * 70)



    save_results(

        metrics,

        predictions,

        targets,

    )





if __name__ == "__main__":

    main()
    
    
    