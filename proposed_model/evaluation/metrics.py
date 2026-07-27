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

        return torch.tensor(0.0)


    return torch.mean(
        torch.abs(
            (prediction-target)
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
        1 -
        ss_residual / ss_total
    )



def calculate_metrics(
    prediction,
    target,
):

    return {

        "MAE":
            mae(
                prediction,
                target
            ).item(),

        "RMSE":
            rmse(
                prediction,
                target
            ).item(),

        "MAPE":
            mape(
                prediction,
                target
            ).item(),

        "R2":
            r2_score(
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
    """
    Calculate all metrics for one batch.

    Parameters
    ----------
    predictions:
        (B,H,N)

    targets:
        (B,H,N)

    Returns
    -------
    dict
        metric values
    """

    if predictions.shape != targets.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"Prediction {predictions.shape}, "
            f"Target {targets.shape}"
        )


    metrics = {

        "MAE":
            mae(
                predictions,
                targets
            ).item(),


        "RMSE":
            rmse(
                predictions,
                targets
            ).item(),


        "MAPE":
            mape(
                predictions,
                targets
            ).item(),


        "R2":
            r2_score(
                predictions,
                targets
            ).item(),

    }


    return metrics



# ==========================================================
# Average Metrics
# ==========================================================


def average_metrics(
    metric_list: list[dict],
):
    """
    Average metrics from multiple batches.
    """

    if len(metric_list) == 0:

        raise ValueError(
            "Metric list is empty."
        )


    keys = metric_list[0].keys()


    averaged = {}


    for key in keys:

        averaged[key] = sum(
            item[key]
            for item in metric_list
        ) / len(metric_list)


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