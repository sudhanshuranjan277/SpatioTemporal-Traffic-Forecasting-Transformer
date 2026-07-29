"""
GraphWaveNet Baseline Training

Output:

comparison_models/graph_wavenet/outputs/

    checkpoints/
        GraphWaveNet_best.pth

    training_history.json
"""


from pathlib import Path
import json
import time


import torch
import torch.nn as nn

from torch.utils.data import DataLoader


from proposed_model.configs.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)


from proposed_model.data.dataset import TrafficDataset


from comparison_models.graph_wavenet.model import (
    GraphWaveNetBaseline
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


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    /
    "GraphWaveNet_best.pth"
)


HISTORY_PATH = (
    OUTPUT_DIR
    /
    "training_history.json"
)



# ======================================================
# Training
# ======================================================


def main():


    print("=" * 70)

    print(
        "GraphWaveNet Baseline Training"
    )

    print("=" * 70)



    # Dataset

    train_dataset = TrafficDataset(
        split="train"
    )


    val_dataset = TrafficDataset(
        split="validation"
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )



    # Model

    model = GraphWaveNetBaseline()


    model.to(
        DEVICE
    )



    criterion = nn.MSELoss()



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )



    best_val_loss = float("inf")



    history = {

        "train_loss": [],

        "val_loss": []

    }



    # Training loop

    for epoch in range(NUM_EPOCHS):


        start = time.time()



        model.train()


        train_loss = 0.0



        for x, y in train_loader:


            x = x.to(
                DEVICE
            )


            y = y.to(
                DEVICE
            )



            optimizer.zero_grad()



            prediction = model(
                x
            )



            loss = criterion(

                prediction,

                y

            )



            loss.backward()


            optimizer.step()



            train_loss += loss.item()



        train_loss /= len(train_loader)



        # Validation

        model.eval()


        val_loss = 0.0



        with torch.no_grad():


            for x, y in val_loader:


                x = x.to(
                    DEVICE
                )


                y = y.to(
                    DEVICE
                )



                prediction = model(
                    x
                )



                loss = criterion(

                    prediction,

                    y

                )



                val_loss += loss.item()



        val_loss /= len(val_loader)



        history["train_loss"].append(
            train_loss
        )


        history["val_loss"].append(
            val_loss
        )



        print(

            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "

            f"Train Loss: {train_loss:.6f} "

            f"Val Loss: {val_loss:.6f} "

            f"Time: {time.time()-start:.2f}s"

        )



        # Save best checkpoint

        if val_loss < best_val_loss:


            best_val_loss = val_loss



            CHECKPOINT_DIR.mkdir(

                parents=True,

                exist_ok=True

            )



            torch.save(

                {

                    "epoch": epoch + 1,

                    "loss": best_val_loss,

                    "model_state_dict":
                    model.state_dict()

                },

                CHECKPOINT_PATH

            )


            print(
                "✓ Best GraphWaveNet checkpoint saved"
            )



    # Save history

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    with open(

        HISTORY_PATH,

        "w"

    ) as f:


        json.dump(

            history,

            f,

            indent=4

        )



    print("=" * 70)

    print(
        "GraphWaveNet Training Completed"
    )


    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )


    print(
        "History:",
        HISTORY_PATH
    )


    print("=" * 70)



if __name__ == "__main__":

    main()