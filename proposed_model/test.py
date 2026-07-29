"""
Traffic Forecasting Research

Final Testing Pipeline

Models:
    - TransGTR
    - LSTM
    - GRU
    - GraphWaveNet


Forecast Horizons:
    - 3 minutes
    - 5 minutes
    - 10 minutes


10 minute forecasting:
    Recursive Multi-Step Forecasting


Metrics:
    MSE
    MAE
    RMSE
    MAPE
    R2

"""


from pathlib import Path
import csv


import torch
import matplotlib.pyplot as plt



# ======================================================
# Imports
# ======================================================


from proposed_model.configs.config import (

    DEVICE,

    OUTPUT_DIR,

    TRANSGTR_CHECKPOINT_DIR

)



from proposed_model.data.evaluation_dataset import (

    EvaluationDataset

)



from proposed_model.models.transgtr import (

    TransGTR

)



from comparison_models.lstm.model import (

    LSTMBaseline

)



from comparison_models.gru.model import (

    GRUBaseline

)



from comparison_models.graph_wavenet.model import (

    GraphWaveNetBaseline

)





# ======================================================
# Project Root
# ======================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]





# ======================================================
# Output Directory
# ======================================================


COMPARISON_DIR = (

    OUTPUT_DIR

    /

    "comparison"

    /

    "combined_dataset"

)





# ======================================================
# Baseline Checkpoints
# ======================================================


LSTM_CHECKPOINT = (

    PROJECT_ROOT

    /

    "comparison_models"

    /

    "lstm"

    /

    "outputs"

    /

    "checkpoints"

    /

    "LSTM_best.pth"

)



GRU_CHECKPOINT = (

    PROJECT_ROOT

    /

    "comparison_models"

    /

    "gru"

    /

    "outputs"

    /

    "checkpoints"

    /

    "GRU_best.pth"

)





GRAPHWAVENET_CHECKPOINT = (

    PROJECT_ROOT

    /

    "comparison_models"

    /

    "graph_wavenet"

    /

    "outputs"

    /

    "checkpoints"

    /

    "GraphWaveNet_best.pth"

)





# ======================================================
# Recursive Forecasting
# ======================================================


def recursive_forecast(

        model,

        x,

        future_steps

):

    """
    Recursive forecasting for
    horizons greater than model output.


    Current model:

        input:
            12 steps

        output:
            5 steps


    Converts:

        5 step model

    into:

        10 step prediction

    """



    predictions = []



    current_input = x.clone()





    while sum(

        p.shape[1]

        for p in predictions

    ) < future_steps:



        output = model(

            current_input

        )



        remaining = (

            future_steps

            -

            sum(

                p.shape[1]

                for p in predictions

            )

        )



        take = min(

            remaining,

            output.shape[1]

        )



        predictions.append(

            output[:, :take, :]

        )



        if remaining <= output.shape[1]:

            break





        # ---------------------------------
        # Update input window
        # ---------------------------------


        batch = current_input.shape[0]

        nodes = current_input.shape[2]

        features = current_input.shape[3]



        next_input = current_input[:, 5:, :, :].clone()



        new_features = torch.zeros(

            batch,

            5,

            nodes,

            features,

            device=x.device

        )



        # Put predicted traffic flow
        # into target feature position


        new_features[:, :, :, 1] = output



        current_input = torch.cat(

            [

                next_input,

                new_features

            ],

            dim=1

        )





    return torch.cat(

        predictions,

        dim=1

    )
# ======================================================
# Metrics
# ======================================================


def calculate_metrics(

        prediction,

        target

):


    prediction = prediction.float()

    target = target.float()



    mse = torch.mean(

        (prediction - target) ** 2

    )



    mae = torch.mean(

        torch.abs(

            prediction - target

        )

    )



    rmse = torch.sqrt(

        mse

    )





    # -----------------------------
    # MAPE
    # -----------------------------


    mask = target > 1



    if torch.sum(mask) > 0:


        mape = torch.mean(

            torch.abs(

                (

                    prediction[mask]

                    -

                    target[mask]

                )

                /

                target[mask]

            )

        ) * 100


    else:


        mape = torch.tensor(0.0)






    # -----------------------------
    # R2
    # -----------------------------


    target_mean = torch.mean(

        target

    )



    denominator = torch.sum(

        (target - target_mean) ** 2

    )



    if denominator == 0:


        r2 = torch.tensor(0.0)


    else:


        r2 = (

            1

            -

            torch.sum(

                (target - prediction) ** 2

            )

            /

            denominator

        )







    return {


        "MSE":

            mse.item(),



        "MAE":

            mae.item(),



        "RMSE":

            rmse.item(),



        "MAPE":

            mape.item(),



        "R2":

            r2.item()


    }







# ======================================================
# Load Model
# ======================================================


def load_model(

        model_name

):



    if model_name == "TransGTR":


        model = TransGTR()



        checkpoint = (

            TRANSGTR_CHECKPOINT_DIR

            /

            "best_model.pth"

        )





    elif model_name == "LSTM":


        model = LSTMBaseline()


        checkpoint = LSTM_CHECKPOINT





    elif model_name == "GRU":


        model = GRUBaseline()


        checkpoint = GRU_CHECKPOINT





    elif model_name == "GraphWaveNet":


        model = GraphWaveNetBaseline()


        checkpoint = GRAPHWAVENET_CHECKPOINT





    else:


        raise ValueError(

            "Unknown Model"

        )





    checkpoint_data = torch.load(

        checkpoint,

        map_location=DEVICE

    )






    if "model_state_dict" in checkpoint_data:


        checkpoint_data = (

            checkpoint_data["model_state_dict"]

        )





    model.load_state_dict(

        checkpoint_data

    )



    model.to(

        DEVICE

    )



    model.eval()



    print(

        f"{model_name} loaded"

    )



    return model







# ======================================================
# Prediction Function
# ======================================================


def get_predictions(

        model,

        loader,

        horizon

):



    predictions = []

    targets = []





    with torch.no_grad():



        for x,y in loader:



            x = x.to(

                DEVICE

            )


            y = y.to(

                DEVICE

            )





            # =================================================
            # Horizon Handling
            # =================================================


            if horizon > 5:


                output = recursive_forecast(

                    model,

                    x,

                    future_steps=horizon

                )



            else:


                output = model(

                    x

                )



                # If horizon is less than model output

                output = output[

                    :,

                    :horizon,

                    :

                ]





            predictions.append(

                output.cpu()

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





    return (

        predictions,

        targets

    )
    
    # ======================================================
# Save Metrics CSV
# ======================================================


def save_metrics(

        results,

        horizon

):


    output_dir = (

        COMPARISON_DIR

        /

        f"horizon_{horizon}min"

    )



    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    file = output_dir / "metrics.csv"




    with open(

        file,

        "w",

        newline=""

    ) as f:



        writer = csv.writer(f)



        writer.writerow(

            [

                "Model",

                "MSE",

                "MAE",

                "RMSE",

                "MAPE",

                "R2"

            ]

        )




        for model,metric in results.items():



            writer.writerow(

                [

                    model,

                    metric["MSE"],

                    metric["MAE"],

                    metric["RMSE"],

                    metric["MAPE"],

                    metric["R2"]

                ]

            )



    print(

        "Saved:",

        file

    )






# ======================================================
# Generate Graph
# ======================================================


def generate_graph(

        results,

        metric,

        horizon

):


    output_dir = (

        COMPARISON_DIR

        /

        f"horizon_{horizon}min"

    )



    names = list(

        results.keys()

    )



    values = [

        results[name][metric]

        for name in names

    ]





    plt.figure(

        figsize=(8,5)

    )



    bars = plt.bar(

        names,

        values

    )





    for bar,value in zip(

            bars,

            values

    ):


        plt.text(

            bar.get_x()

            +

            bar.get_width()/2,


            bar.get_height(),


            f"{value:.4f}",


            ha="center",

            va="bottom"

        )





    plt.title(

        f"{metric} Comparison - {horizon} Minute Forecast"

    )


    plt.ylabel(

        metric

    )


    plt.xticks(

        rotation=30

    )


    plt.grid(

        axis="y",

        alpha=0.3

    )



    plt.tight_layout()



    plt.savefig(

        output_dir

        /

        f"{metric.lower()}.png",

        dpi=300

    )


    plt.close()







# ======================================================
# Main Testing Pipeline
# ======================================================


def main():


    print("="*70)

    print(

        "Multi Horizon Traffic Forecasting Testing"

    )

    print("="*70)





    models = [


        "TransGTR",

        "LSTM",

        "GRU",

        "GraphWaveNet"


    ]





    loaded_models = {}



    for name in models:


        loaded_models[name] = load_model(

            name

        )






    # ==============================================
    # Test Horizons
    # ==============================================


    horizons = [

        3,

        5,

        10

    ]





    for horizon in horizons:



        print()

        print("#"*70)

        print(

            f"Testing Horizon: {horizon} minutes"

        )

        print("#"*70)





        dataset = EvaluationDataset(

            horizon=horizon

        )





        print(

            "Dataset Samples:",

            len(dataset)

        )





        loader = torch.utils.data.DataLoader(

            dataset,

            batch_size=32,

            shuffle=False

        )





        results = {}
        for name,model in loaded_models.items():



            print()

            print(

                f"Evaluating: {name}"

            )
            prediction,target = get_predictions(

                model,

                loader,

                horizon

            )
            print(
                "Prediction shape:",
                prediction.shape
            )
            print(
                "Target shape:",target.shape)

            results[name] = calculate_metrics(

                prediction,

                target

            )
            print(
                name,
                results[name]
                )
            
            
            print("\nFINAL RESULTS")
            print("="*70)
            for model_name, metric in results.items():
                print(model_name)
                
                print("MSE :", metric["MSE"])
                print("MAE :", metric["MAE"])
                print("RMSE:", metric["RMSE"])
                print("MAPE:", metric["MAPE"])
                print("R2  :", metric["R2"])

                print("-"*70)






        # Save table

        save_metrics(

            results,

            horizon

        )






        # Generate graphs


        for metric in [

            "MSE",

            "MAE",

            "RMSE"

        ]:


            generate_graph(

                results,

                metric,

                horizon

            )







    print()

    print("="*70)

    print(

        "Testing Completed Successfully"

    )

    print("="*70)







# ======================================================
# Run
# ======================================================


if __name__ == "__main__":


    main()
