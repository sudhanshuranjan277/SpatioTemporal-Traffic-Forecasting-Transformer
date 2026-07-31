"""
LSTM Horizon Experiments Runner

Runs:

3 min
5 min
8 min

Training + Evaluation automatically
"""


from comparison_models.lstm.train import train

from comparison_models.lstm.evaluate import evaluate





# ======================================================
# Horizons
# ======================================================


HORIZONS = [

    3,

    5,

    8

]





def main():


    print("="*70)

    print(
        "LSTM Multi Horizon Experiment"
    )

    print("="*70)



    all_results = {}



    for horizon in HORIZONS:


        print()

        print("#"*70)

        print(
            f"STARTING HORIZON {horizon} MIN"
        )

        print("#"*70)



        # -------------------------
        # Training
        # -------------------------


        train(

            horizon

        )



        # -------------------------
        # Evaluation
        # -------------------------


        result = evaluate(

            horizon

        )



        all_results[

            f"{horizon}_min"

        ] = result




    print()

    print("="*70)

    print(
        "ALL HORIZON EXPERIMENTS COMPLETED"
    )

    print("="*70)



    for h,result in all_results.items():

        print()

        print(h)

        for key,value in result.items():

            print(

                f"{key}: {value:.6f}"

            )



    print("="*70)






if __name__ == "__main__":

    main()