"""
Experiment Runner

Traffic Forecasting Research

Runs multi-horizon experiments:

Forecast Horizon:
    - 3 minutes
    - 5 minutes
    - 10 minutes

Models:
    - LSTM
    - GRU
    - GraphWaveNet
    - TransGTR


Outputs:

outputs/comparison/

    horizon_3min/
        metrics.csv

    horizon_5min/
        metrics.csv

    horizon_10min/
        metrics.csv

"""

from pathlib import Path
import subprocess
import json
import csv
import shutil


# ==========================================================
# Paths
# ==========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "proposed_model"
    /
    "outputs"
    /
    "comparison"
)


CONFIG_FILE = (
    PROJECT_ROOT
    /
    "proposed_model"
    /
    "configs"
    /
    "config.py"
)



# ==========================================================
# Experiment Settings
# ==========================================================


HORIZONS = [
    3,
    5,
    10
]


MODELS = [

    "LSTM",

    "GRU",

    "GraphWaveNet",

    "TransGTR"

]



# ==========================================================
# Update Prediction Horizon
# ==========================================================


def update_prediction_horizon(
        horizon:int
):

    """
    Dynamically update config horizon.
    """


    text = CONFIG_FILE.read_text()


    old = (
        "PREDICTION_HORIZON = "
        + 
        str(
            3
        )
    )


    new = (
        "PREDICTION_HORIZON = "
        +
        str(horizon)
    )


    text = text.replace(
        old,
        new
    )


    CONFIG_FILE.write_text(
        text
    )


    print(
        f"Prediction horizon updated: {horizon} minutes"
    )



# ==========================================================
# Run Model Training / Testing
# ==========================================================


def run_model(
        model_name
):


    print(
        "\n"
        +
        "="*70
    )


    print(
        f"Running {model_name}"
    )


    print(
        "="*70
    )


    if model_name == "TransGTR":


        subprocess.run(

            [
                "python",
                "-m",
                "proposed_model.train"
            ],

            check=True

        )


        subprocess.run(

            [
                "python",
                "-m",
                "proposed_model.evaluation.evaluate"
            ],

            check=True

        )



    elif model_name in [

        "LSTM",

        "GRU",

        "GraphWaveNet"

    ]:


        subprocess.run(

            [
                "python",
                "-m",
                "comparison_models.train_baselines"
            ],

            check=True

        )



    else:


        raise ValueError(
            f"Unknown model {model_name}"
        )



# ==========================================================
# Collect Metrics
# ==========================================================


def collect_metrics():


    results = {}



    # TransGTR

    transgtr_file = (

        PROJECT_ROOT

        /

        "proposed_model"

        /

        "outputs"

        /

        "test_metrics.json"

    )



    if transgtr_file.exists():

        with open(
            transgtr_file
        ) as f:

            results["TransGTR"] = json.load(f)



    # Baselines

    baseline_file = (

        PROJECT_ROOT

        /

        "proposed_model"

        /

        "outputs"

        /

        "baseline_results.json"

    )



    if baseline_file.exists():

        with open(
            baseline_file
        ) as f:

            baseline_results = json.load(f)



        results.update(
            baseline_results
        )



    return results




# ==========================================================
# Save CSV
# ==========================================================


def save_results(
        horizon,
        results
):


    output_folder = (

        OUTPUT_DIR

        /

        f"horizon_{horizon}min"

    )


    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )



    csv_file = (

        output_folder

        /

        "metrics.csv"

    )



    with open(

        csv_file,

        "w",

        newline=""

    ) as f:


        writer = csv.writer(f)


        writer.writerow(

            [

                "Model",

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

                    metric.get(
                        "MAE",
                        ""
                    ),

                    metric.get(
                        "RMSE",
                        ""
                    ),

                    metric.get(
                        "MAPE",
                        ""
                    ),

                    metric.get(
                        "R2",
                        ""
                    )

                ]

            )



    print(
        f"Saved: {csv_file}"
    )



# ==========================================================
# Main Experiment
# ==========================================================


def main():


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    for horizon in HORIZONS:


        print("\n")

        print(
            "#"*80
        )

        print(
            f"STARTING {horizon} MINUTE FORECAST EXPERIMENT"
        )

        print(
            "#"*80
        )



        update_prediction_horizon(
            horizon
        )



        for model in MODELS:


            run_model(
                model
            )



        results = collect_metrics()



        save_results(

            horizon,

            results

        )



    print(

        "\nAll experiments completed successfully."

    )



# ==========================================================


if __name__ == "__main__":

    main()