"""
analysis/model_summary.py

TransGTR Model Analysis

Responsibilities
-----------------
1. Build model
2. Count parameters
3. Display architecture details
4. Verify input/output shapes
"""


from __future__ import annotations


import torch


from models.transgtr import TransGTR


from configs.config import (
    HISTORY_LENGTH,
    NUM_INPUT_FEATURES,
    PREDICTION_HORIZON,
)





# ==========================================================
# Parameter Counting
# ==========================================================


def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    return total, trainable





# ==========================================================
# Component Parameters
# ==========================================================


def component_summary(model):

    print("=" * 70)

    print(
        "Component Parameter Summary"
    )

    print("=" * 70)



    components = {

        "Traffic Embedding":
            model.embedding,


        "TSFormer":
            model.temporal_encoder,


        "Dynamic Graph Generator":
            model.graph_generator,


        "Graph WaveNet":
            model.graph_wavenet,


        "Prediction Head":
            model.prediction_head,

    }



    for name, module in components.items():

        params = sum(
            p.numel()
            for p in module.parameters()
        )


        print(
            f"{name:<30}: {params:,}"
        )


    print("=" * 70)
    
    # ==========================================================
# Model Summary
# ==========================================================


def model_summary():


    print("=" * 70)

    print(
        "TransGTR Model Summary"
    )

    print("=" * 70)



    # ----------------------------------------------
    # Build Model
    # ----------------------------------------------

    model = TransGTR()



    total_params, trainable_params = (
        count_parameters(model)
    )



    # ----------------------------------------------
    # Component Summary
    # ----------------------------------------------

    component_summary(
        model
    )



    # ----------------------------------------------
    # Overall Parameters
    # ----------------------------------------------

    print()

    print(
        "Total Parameters     : "
        f"{total_params:,}"
    )


    print(
        "Trainable Parameters : "
        f"{trainable_params:,}"
    )


    print("=" * 70)



    # ----------------------------------------------
    # Input / Output Shape Test
    # ----------------------------------------------

    batch_size = 1

    num_nodes = 9



    dummy_input = torch.randn(

        batch_size,

        HISTORY_LENGTH,

        num_nodes,

        NUM_INPUT_FEATURES,

    )



    with torch.no_grad():

        output = model(
            dummy_input
        )



    print(
        "Input Shape  :",
        dummy_input.shape
    )


    print(
        "Output Shape :",
        output.shape
    )


    print("=" * 70)



    expected_output = (

        batch_size,

        PREDICTION_HORIZON,

        num_nodes,

    )


    assert output.shape == expected_output, (

        f"Expected {expected_output}, "
        f"got {output.shape}"

    )


    print(
        "✓ Output shape verification passed"
    )


    print("=" * 70)





# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":

    model_summary()