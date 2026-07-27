"""
train_baselines.py

Training Pipeline for Baseline Models

Models
------
1. LSTM
2. GRU
3. GraphWaveNet


Uses same:
- Dataset
- Loss
- Optimizer
- Evaluation pipeline

as TransGTR
"""


from __future__ import annotations


import time
import json
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader



from proposed_model.configs.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    OUTPUT_DIR,
)



from proposed_model.data.dataset import (
    TrafficDataset,
)



from proposed_model.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
)



from comparison_models.lstm.model import (
    LSTMBaseline,
)


from comparison_models.gru.model import (
    GRUBaseline,
)


from comparison_models.graph_wavenet.model import (
    GraphWaveNetBaseline,
)





# ==========================================================
# Device
# ==========================================================


device = DEVICE





# ==========================================================
# Dataset
# ==========================================================


def create_dataloaders():


    train_dataset = TrafficDataset(

        split="train"

    )


    validation_dataset = TrafficDataset(

        split="validation"

    )


    test_dataset = TrafficDataset(

        split="test"

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

    )


    return (
        train_loader,
        validation_loader,
        test_loader,
    )
    
    # ==========================================================
# Model Factory
# ==========================================================


def build_model(
    model_name: str,
):


    if model_name == "LSTM":

        model = LSTMBaseline()



    elif model_name == "GRU":

        model = GRUBaseline()



    elif model_name == "GraphWaveNet":

        model = GraphWaveNetBaseline()



    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )



    return model.to(device)





# ==========================================================
# Training One Epoch
# ==========================================================


def train_one_epoch(

    model,

    loader,

    criterion,

    optimizer,

):


    model.train()


    total_loss = 0.0



    for x, y in loader:


        x = x.to(device)

        y = y.to(device)



        optimizer.zero_grad()



        prediction = model(
            x
        )


        loss = criterion(

            prediction,

            y,

        )


        loss.backward()



        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0,

        )


        optimizer.step()



        total_loss += loss.item()



    return total_loss / len(loader)





# ==========================================================
# Validation
# ==========================================================


@torch.no_grad()
def validate(

    model,

    loader,

    criterion,

):


    model.eval()


    total_loss = 0.0



    for x, y in loader:


        x = x.to(device)

        y = y.to(device)



        prediction = model(
            x
        )



        loss = criterion(

            prediction,

            y,

        )



        total_loss += loss.item()



    return total_loss / len(loader)





# ==========================================================
# Train Model
# ==========================================================


def train_model(

    model_name,

):


    print("=" * 70)

    print(
        f"Training {model_name}"
    )

    print("=" * 70)



    train_loader, validation_loader, _ = (
        create_dataloaders()
    )



    model = build_model(
        model_name
    )



    criterion = nn.MSELoss()



    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )



    best_loss = float("inf")



    history = {

        "train_loss": [],

        "validation_loss": [],

    }



    checkpoint_dir = (

        OUTPUT_DIR /

        "baseline_checkpoints"

    )


    checkpoint_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    for epoch in range(NUM_EPOCHS):


        start = time.time()



        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

        )



        val_loss = validate(

            model,

            validation_loader,

            criterion,

        )



        history["train_loss"].append(
            train_loss
        )


        history["validation_loss"].append(
            val_loss
        )



        if val_loss < best_loss:


            best_loss = val_loss



            torch.save(

                {

                "model_state_dict":
                    model.state_dict(),

                "loss":
                    best_loss,

                "epoch":
                    epoch + 1,

                },

                checkpoint_dir /

                f"{model_name}_best.pth"

            )



        print(

            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "

            f"Train: {train_loss:.6f} "

            f"Val: {val_loss:.6f} "

            f"Time: {time.time()-start:.2f}s"

        )


    return history

# ==========================================================
# Evaluate Baseline Model
# ==========================================================


@torch.no_grad()
def evaluate_model(

    model_name,

):


    print("=" * 70)

    print(
        f"Evaluating {model_name}"
    )

    print("=" * 70)



    _, _, test_loader = (
        create_dataloaders()
    )



    model = build_model(
        model_name
    )



    checkpoint_path = (

        OUTPUT_DIR /

        "baseline_checkpoints" /

        f"{model_name}_best.pth"

    )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

    )



    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )



    model.eval()



    predictions = []

    targets = []



    for x, y in test_loader:


        x = x.to(device)



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
        predictions,
        dim=0
    )


    targets = torch.cat(
        targets,
        dim=0
    )



    metrics = {


        "MAE":

            calculate_mae(

                predictions,

                targets

            ).item(),



        "RMSE":

            calculate_rmse(

                predictions,

                targets

            ).item(),



        "MAPE":

            calculate_mape(

                predictions,

                targets

            ).item(),



        "R2":

            calculate_r2(

                predictions,

                targets

            ).item(),


    }



    print(metrics)


    return metrics





# ==========================================================
# Run All Baselines
# ==========================================================


def main():


    models = [

        "LSTM",

        "GRU",

        "GraphWaveNet",

    ]



    results = {}



    for model_name in models:


        train_model(
            model_name
        )


        results[model_name] = evaluate_model(

            model_name

        )



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    result_path = (

        OUTPUT_DIR /

        "baseline_results.json"

    )



    with open(

        result_path,

        "w"

    ) as file:


        json.dump(

            results,

            file,

            indent=4,

        )



    print("=" * 70)

    print(
        "Baseline Comparison Completed"
    )


    print(
        f"Results Saved: {result_path}"
    )

    print("=" * 70)





if __name__ == "__main__":

    main()
