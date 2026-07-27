"""
analysis/inference_test.py

TransGTR Inference Performance Test

Responsibilities
----------------
1. Load trained model checkpoint.
2. Run inference on test samples.
3. Measure inference time.
4. Verify output shape.
"""


from __future__ import annotations


import time


import torch


from torch.utils.data import DataLoader



from configs.config import (
    DEVICE,
    CHECKPOINT_DIR,
    BATCH_SIZE,
)


from models.transgtr import TransGTR


from data.dataset import TrafficDataset





# ==========================================================
# Load Model
# ==========================================================


def load_model():


    model = TransGTR()


    model = model.to(
        DEVICE
    )



    checkpoint_path = (
        CHECKPOINT_DIR /
        "best_model.pth"
    )



    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE

    )



    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model.eval()



    print("=" * 70)

    print(
        "Inference Model Loaded"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Epoch: {checkpoint['epoch']}"
    )

    print("=" * 70)



    return model





# ==========================================================
# Test Data Loader
# ==========================================================


def create_test_loader():


    dataset = TrafficDataset(

        split="test"

    )



    loader = DataLoader(

        dataset,

        batch_size=1,

        shuffle=False,

        num_workers=0,

    )


    return loader

# ==========================================================
# Inference Benchmark
# ==========================================================


@torch.no_grad()
def run_inference_test(
    model,
    test_loader,
):


    inference_times = []


    sample_count = 0



    print("=" * 70)

    print(
        "Running Inference Test"
    )

    print("=" * 70)



    for batch in test_loader:


        x, y = batch



        x = x.to(
            DEVICE
        )



        # ------------------------------------------
        # Measure latency
        # ------------------------------------------


        start_time = time.perf_counter()



        prediction = model(
            x
        )



        end_time = time.perf_counter()



        elapsed = (
            end_time -
            start_time
        )



        inference_times.append(
            elapsed
        )



        sample_count += x.size(0)



        print(
            "Input Shape  :",
            x.shape
        )


        print(
            "Output Shape :",
            prediction.shape
        )



        break



    average_time = (

        sum(inference_times)

        /

        len(inference_times)

    )



    latency_ms = (
        average_time * 1000
    )



    print("=" * 70)

    print(
        "Inference Results"
    )

    print("=" * 70)



    print(
        f"Samples Tested : {sample_count}"
    )


    print(
        f"Latency        : {latency_ms:.4f} ms"
    )


    print("=" * 70)



    return latency_ms





# ==========================================================
# Main
# ==========================================================


def main():


    print("=" * 70)

    print(
        "TransGTR Inference Test"
    )

    print("=" * 70)



    model = load_model()


    test_loader = create_test_loader()



    run_inference_test(

        model,

        test_loader,

    )



    print(
        "✓ Inference test completed"
    )

    print("=" * 70)





if __name__ == "__main__":

    main()