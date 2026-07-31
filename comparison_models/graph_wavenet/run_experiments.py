"""
Graph WaveNet Multi Horizon Experiment Runner


Runs:

3 minutes
5 minutes
8 minutes


Pipeline:

Training
    |
    ↓
Checkpoint Saving


Evaluation will be added after evaluate.py


"""


from comparison_models.graph_wavenet.train import train





# ======================================================
# Horizons
# ======================================================


HORIZONS = [

    3,

    5,

    8

]





# ======================================================
# Main
# ======================================================


def main():


    print("="*70)

    print(

        "Graph WaveNet Multi Horizon Experiment"

    )

    print("="*70)





    for horizon in HORIZONS:



        print()


        print("#"*70)


        print(

            f"STARTING HORIZON {horizon} MIN"

        )


        print("#"*70)





        # -----------------------------
        # Training
        # -----------------------------


        train(

            horizon

        )





    print()


    print("="*70)

    print(

        "Graph WaveNet Experiments Completed"

    )

    print("="*70)





# ======================================================
# Run
# ======================================================


if __name__ == "__main__":


    main()