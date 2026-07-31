"""
GRU Multi Horizon Experiment Runner

Runs complete GRU experiments for:

3 minutes
5 minutes
8 minutes


Pipeline:

Training
   |
   ↓
Checkpoint Saving
   |
   ↓
Evaluation
   |
   ↓
Metrics Saving
"""


from comparison_models.gru.train import train

from comparison_models.gru.evaluate import evaluate





# ======================================================
# Forecast Horizons
# ======================================================


HORIZONS = [

    3,

    5,

    8

]





# ======================================================
# Main Experiment
# ======================================================


def main():


    print("=" * 70)

    print(
        "GRU Multi Horizon Experiment"
    )

    print("=" * 70)




    for horizon in HORIZONS:


        print()

        print("#" * 70)

        print(

            f"STARTING HORIZON {horizon} MIN"

        )

        print("#" * 70)



        # -----------------------------
        # Training
        # -----------------------------


        train(

            horizon

        )



        # -----------------------------
        # Evaluation
        # -----------------------------


        evaluate(

            horizon

        )





    print()

    print("=" * 70)

    print(

        "GRU ALL EXPERIMENTS COMPLETED"

    )

    print("=" * 70)





# ======================================================
# Run
# ======================================================


if __name__ == "__main__":

    main()