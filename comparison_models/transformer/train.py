"""
Transformer Training Module

Traffic Forecasting

Input:
Previous traffic observations

Output:
Future traffic flow prediction


Horizons:

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
# Hyperparameters
# ======================================================


BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001






# ======================================================
# Paths
# ======================================================


CURRENT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = CURRENT_DIR / "outputs"


CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

HISTORY_DIR = OUTPUT_DIR / "history"



CHECKPOINT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


HISTORY_DIR.mkdir(

    parents=True,

    exist_ok=True

)







# ======================================================
# Training Function
# ======================================================


def train(horizon):


    print("="*70)

    print(

        f"Transformer Training | Horizon {horizon} min"

    )

    print("="*70)




    # -----------------------------
    # Dataset
    # -----------------------------


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







    # -----------------------------
    # Model
    # -----------------------------


    model = TrafficTransformer(

        horizon=horizon

    )



    model.to(DEVICE)






    criterion = nn.MSELoss()



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )





    best_loss = float("inf")





    history = {

        "train_loss": [],

        "validation_loss": []

    }







    # ==================================================
    # Epoch Loop
    # ==================================================


    for epoch in range(EPOCHS):


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







        # -----------------------------
        # Validation
        # -----------------------------


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


        history["validation_loss"].append(

            val_loss

        )






        print(

            f"Epoch [{epoch+1}/{EPOCHS}] "

            f"Train Loss: {train_loss:.6f} "

            f"Val Loss: {val_loss:.6f} "

            f"Time: {time.time()-start:.2f}s"

        )







        # -----------------------------
        # Save Best Model
        # -----------------------------


        if val_loss < best_loss:



            best_loss = val_loss



            checkpoint = (

                CHECKPOINT_DIR

                /

                f"Transformer_{horizon}min.pth"

            )



            torch.save(

                {

                    "epoch":epoch+1,

                    "horizon":horizon,

                    "model_state_dict":

                        model.state_dict(),

                    "loss":best_loss

                },

                checkpoint

            )



            print(

                "✓ Checkpoint Saved"

            )









    # -----------------------------
    # Save History
    # -----------------------------


    history_file = (

        HISTORY_DIR

        /

        f"Transformer_{horizon}min_history.json"

    )




    with open(

        history_file,

        "w"

    ) as f:


        json.dump(

            history,

            f,

            indent=4

        )







    print()

    print("="*70)

    print(

        f"Transformer {horizon} min Training Completed"

    )


    print(

        "Checkpoint:",

        CHECKPOINT_DIR /

        f"Transformer_{horizon}min.pth"

    )


    print("="*70)






# ======================================================
# Main
# ======================================================


if __name__ == "__main__":


    train(5)