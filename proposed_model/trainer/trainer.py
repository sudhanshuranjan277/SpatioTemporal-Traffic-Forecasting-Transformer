"""
trainer/trainer.py

Training Engine for TransGTR

Pipeline

Dataset
   |
DataLoader
   |
TransGTR
   |
Loss
   |
Optimizer
   |
Checkpoint
"""

from __future__ import annotations


import time
import random
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader



# ==========================================================
# Project Imports
# ==========================================================


from configs.config import (

    BATCH_SIZE,

    NUM_EPOCHS,

    LEARNING_RATE,

    WEIGHT_DECAY,

    DEVICE,

    RANDOM_SEED,

    CHECKPOINT_DIR,

)


from data.dataset import TrafficDataset


from models.transgtr import TransGTR





# ==========================================================
# Device
# ==========================================================


device = DEVICE



# ==========================================================
# Reproducibility
# ==========================================================


def set_seed(
    seed: int = RANDOM_SEED,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    
    
    # ==========================================================
# DataLoader Creation
# ==========================================================


def create_dataloaders():

    train_dataset = TrafficDataset(
        split="train"
    )


    validation_dataset = TrafficDataset(
        split="validation"
    )


    train_loader = DataLoader(

        dataset=train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        drop_last=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

    )


    validation_loader = DataLoader(

        dataset=validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

    )


    return (
        train_loader,
        validation_loader,
    )



# ==========================================================
# Model
# ==========================================================


def build_model():

    model = TransGTR()

    model = model.to(device)


    return model




# ==========================================================
# Optimizer
# ==========================================================


def build_optimizer(
    model,
):

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )


    return optimizer




# ==========================================================
# Loss
# ==========================================================


def build_loss():


    return nn.MSELoss()

# ==========================================================
# Scheduler
# ==========================================================


def build_scheduler(
    optimizer,
):

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=5,

    )

    return scheduler



# ==========================================================
# Checkpoint Saving
# ==========================================================


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    filename,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    checkpoint_path = (
        CHECKPOINT_DIR / filename
    )


    torch.save(

        {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss": loss,

        },

        checkpoint_path,

    )



    print(
        f"Checkpoint saved : {checkpoint_path}"
    )




# ==========================================================
# NaN / Inf Check
# ==========================================================


def check_tensor(
    tensor,
    name,
):

    if torch.isnan(tensor).any():

        raise RuntimeError(
            f"NaN detected in {name}"
        )


    if torch.isinf(tensor).any():

        raise RuntimeError(
            f"Infinity detected in {name}"
        )



# ==========================================================
# Train One Epoch
# ==========================================================


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()


    total_loss = 0.0

    total_samples = 0



    for batch_idx, batch in enumerate(loader):


        x, y = batch


        x = x.to(device)

        y = y.to(device)



        check_tensor(
            x,
            "Input"
        )


        check_tensor(
            y,
            "Target"
        )



        optimizer.zero_grad()



        prediction = model(x)



        check_tensor(
            prediction,
            "Prediction"
        )



        loss = criterion(

            prediction,

            y,

        )



        check_tensor(
            loss,
            "Loss"
        )



        loss.backward()



        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0,

        )



        optimizer.step()



        batch_size = x.size(0)


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total_samples += batch_size



    return (
        total_loss
        /
        total_samples
    )    
    
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

    total_samples = 0



    for batch in loader:


        x, y = batch


        x = x.to(device)

        y = y.to(device)



        check_tensor(
            x,
            "Validation Input"
        )


        check_tensor(
            y,
            "Validation Target"
        )



        prediction = model(x)



        check_tensor(
            prediction,
            "Validation Prediction"
        )



        loss = criterion(

            prediction,

            y,

        )



        check_tensor(
            loss,
            "Validation Loss"
        )



        batch_size = x.size(0)



        total_loss += (

            loss.item()

            *

            batch_size

        )


        total_samples += batch_size



    return (

        total_loss

        /

        total_samples

    )




# ==========================================================
# Training Pipeline
# ==========================================================


def fit():

    print("=" * 70)

    print("TransGTR Training")

    print("=" * 70)



    set_seed()



    # -------------------------------
    # Data
    train_loader, validation_loader = create_dataloaders()
    
    train_loader, validation_loader = (
        create_dataloaders()
    )



    # -------------------------------
    # Model
    # -------------------------------


    model = build_model()



    criterion = build_loss()



    optimizer = build_optimizer(
        model
    )


    scheduler = build_scheduler(
        optimizer
    )



    best_validation_loss = float(
        "inf"
    )



    print(
        f"Device : {device}"
    )


    print(
        f"Train Samples : "
        f"{len(train_loader.dataset)}"
    )


    print(
        f"Validation Samples : "
        f"{len(validation_loader.dataset)}"
    )


    print("=" * 70)




    history = {

        "train_loss": [],

        "validation_loss": [],

    }



    # -------------------------------
    # Epoch Loop
    # -------------------------------


    for epoch in range(
        NUM_EPOCHS
    ):


        start_time = time.time()



        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

        )



        validation_loss = validate(

            model,

            validation_loader,

            criterion,

        )



        scheduler.step(
            validation_loss
        )



        history["train_loss"].append(
            train_loss
        )


        history["validation_loss"].append(
            validation_loss
        )



        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )



        elapsed = (
            time.time()
            -
            start_time
        )



        print(

            f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d}] "

            f"| Train Loss: {train_loss:.6f} "

            f"| Val Loss: {validation_loss:.6f} "

            f"| LR: {current_lr:.6e} "

            f"| Time: {elapsed:.2f}s"

        )



        # -------------------------------
        # Save Best
        # -------------------------------


        if validation_loss < best_validation_loss:


            best_validation_loss = (
                validation_loss
            )


            save_checkpoint(

                model,

                optimizer,

                epoch + 1,

                best_validation_loss,

                "best_model.pth",

            )



        # Save latest model

        save_checkpoint(

            model,

            optimizer,

            epoch + 1,

            validation_loss,

            "last_model.pth",

        )



    print("=" * 70)

    print("Training Completed")

    print("=" * 70)
    
     # SAVE HISTORY HERE
    # ==============================

    import json

    from configs.config import OUTPUT_DIR


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    history_path = OUTPUT_DIR / "training_history.json"


    with open(history_path, "w") as file:

        json.dump(
            history,
            file,
            indent=4
        )


    print(
        f"Training history saved: {history_path}"
    )
    
    return history

# ==========================================================
# Main Entry Point
# ==========================================================


def main():

    history = fit()


    print()

    print("=" * 70)

    print("Training History")

    print("=" * 70)


    print(
        "Final Train Loss :",
        history["train_loss"][-1]
    )


    print(
        "Final Validation Loss :",
        history["validation_loss"][-1]
    )


    print("=" * 70)



# ==========================================================
# Run
# ==========================================================


if __name__ == "__main__":

    main()