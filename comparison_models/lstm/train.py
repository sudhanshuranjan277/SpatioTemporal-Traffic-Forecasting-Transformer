"""
LSTM Baseline Training

Dynamic Horizon Training

Supports:
3 min
5 min
8 min
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


from comparison_models.lstm.model import LSTMBaseline





# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CHECKPOINT_DIR = (
    OUTPUT_DIR /
    "checkpoints"
)





# ======================================================
# Training Function
# ======================================================


def train(horizon):


    print("="*70)

    print(
        f"LSTM Training | Horizon {horizon} min"
    )

    print("="*70)




    checkpoint_path = (

        CHECKPOINT_DIR
        /
        f"LSTM_{horizon}min.pth"

    )



    history_path = (

        OUTPUT_DIR
        /
        f"LSTM_{horizon}min_history.json"

    )





    # ==================================================
    # Dataset
    # ==================================================


    train_dataset = TrafficDataset(

        split="train",

        horizon=horizon

    )



    val_dataset = TrafficDataset(

        split="validation",

        horizon=horizon

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





    print()

    print(
        "Train Samples:",
        len(train_dataset)
    )

    print(
        "Validation Samples:",
        len(val_dataset)
    )





    # ==================================================
    # Model
    # ==================================================


    model = LSTMBaseline(

        horizon=horizon

    )


    model.to(DEVICE)





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





    # ==================================================
    # Training Loop
    # ==================================================


    for epoch in range(NUM_EPOCHS):


        start = time.time()



        model.train()


        train_loss = 0



        for x,y in train_loader:



            x = x.to(DEVICE)

            y = y.to(DEVICE)



            optimizer.zero_grad()



            prediction = model(x)



            loss = criterion(

                prediction,

                y

            )



            loss.backward()



            optimizer.step()



            train_loss += loss.item()





        train_loss /= len(train_loader)





        # ==================================================
        # Validation
        # ==================================================


        model.eval()


        val_loss = 0



        with torch.no_grad():


            for x,y in val_loader:



                x = x.to(DEVICE)

                y = y.to(DEVICE)



                prediction = model(x)



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

            f"Train: {train_loss:.4f} "

            f"Val: {val_loss:.4f} "

            f"Time: {time.time()-start:.2f}s"

        )







        # ==================================================
        # Save Best Model
        # ==================================================


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

                    "horizon": horizon,

                    "model_state_dict":
                    model.state_dict()

                },

                checkpoint_path

            )



            print(

                "✓ Checkpoint saved"

            )







    # ==================================================
    # Save History
    # ==================================================


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    with open(

        history_path,

        "w"

    ) as f:



        json.dump(

            history,

            f,

            indent=4

        )





    print("="*70)

    print(

        f"LSTM {horizon} min Training Completed"

    )



    print(

        "Checkpoint:",

        checkpoint_path

    )


    print(

        "History:",

        history_path

    )


    print("="*70)



    return checkpoint_path





# ======================================================
# Direct Run
# ======================================================


if __name__ == "__main__":


    train(5)