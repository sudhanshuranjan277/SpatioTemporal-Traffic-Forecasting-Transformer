"""
evaluation/metrics.py

Traffic Forecasting Evaluation Metrics

Metrics:
- MAE
- RMSE
- MAPE
- R2 Score
"""


from __future__ import annotations


import torch





# ==========================================================
# Basic Metrics
# ==========================================================


def mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    return torch.mean(

        torch.abs(

            prediction - target

        )

    )





def rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    mse = torch.mean(

        (prediction - target) ** 2

    )

    return torch.sqrt(mse)





def mape(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 1.0,
):


    mask = target.abs() > threshold



    prediction = prediction[mask]

    target = target[mask]



    if target.numel() == 0:

        return torch.tensor(
            0.0
        )



    return torch.mean(

        torch.abs(

            (prediction - target)

            /

            target

        )

    ) * 100





def r2_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
):


    target_mean = torch.mean(
        target
    )


    ss_total = torch.sum(

        (target - target_mean) ** 2

    )


    ss_residual = torch.sum(

        (target - prediction) ** 2

    )


    return (

        1

        -

        ss_residual / ss_total

    )





# ==========================================================
# Compatibility Functions
# Used by comparison_models/train_baselines.py
# ==========================================================


def calculate_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    return mae(
        prediction,
        target
    )





def calculate_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    return rmse(
        prediction,
        target
    )





def calculate_mape(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    return mape(
        prediction,
        target
    )





def calculate_r2(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    return r2_score(
        prediction,
        target
    )





# ==========================================================
# Complete Metrics
# ==========================================================


def calculate_metrics(
    prediction,
    target,
):


    return {


        "MAE":

            calculate_mae(

                prediction,

                target

            ).item(),



        "RMSE":

            calculate_rmse(

                prediction,

                target

            ).item(),



        "MAPE":

            calculate_mape(

                prediction,

                target

            ).item(),



        "R2":

            calculate_r2(

                prediction,

                target

            ).item(),

    }





# ==========================================================
# Batch Metric Evaluation
# ==========================================================


def evaluate_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
):


    if predictions.shape != targets.shape:

        raise ValueError(

            f"Shape mismatch: "
            f"Prediction {predictions.shape}, "
            f"Target {targets.shape}"

        )



    return calculate_metrics(

        predictions,

        targets

    )





# ==========================================================
# Average Metrics
# ==========================================================


def average_metrics(
    metric_list: list[dict],
):


    if len(metric_list) == 0:

        raise ValueError(

            "Metric list is empty."

        )



    keys = metric_list[0].keys()



    averaged = {}



    for key in keys:


        averaged[key] = (

            sum(

                item[key]

                for item in metric_list

            )

            /

            len(metric_list)

        )



    return averaged





# ==========================================================
# Unit Test
# ==========================================================


if __name__ == "__main__":


    batch = 4

    horizon = 3

    nodes = 9



    prediction = torch.randn(

        batch,

        horizon,

        nodes,

    )



    target = torch.randn(

        batch,

        horizon,

        nodes,

    )



    results = evaluate_batch(

        prediction,

        target,

    )



    print("=" * 60)

    print(
        "Metrics Test"
    )

    print("=" * 60)



    for name, value in results.items():

        print(

            f"{name}: {value:.6f}"

        )



    print("=" * 60)

    print(
        "✓ Metrics module working"
    )

    print("=" * 60)