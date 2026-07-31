"""
Transformer Multi Horizon Training

Runs:

3 min
5 min
8 min

"""


from comparison_models.transformer.train import train





def main():


    horizons = [

        3,

        5,

        8

    ]



    for horizon in horizons:


        print()

        print("="*70)

        print(

            f"Running Transformer Experiment: {horizon} min"

        )

        print("="*70)



        train(horizon)





if __name__ == "__main__":


    main()