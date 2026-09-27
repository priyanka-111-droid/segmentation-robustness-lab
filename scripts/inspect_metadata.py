import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.metadata)

    print("\nDataset size:")
    print(len(df))

    print("\nObject statistics:")
    print(
        df[
            [
                "num_objects",
                "num_small",
                "num_medium",
                "num_large",
                "small_fraction",
                "medium_fraction",
                "large_fraction",
            ]
        ].describe()
    )

    print("\nImages by small-object fraction:")
    print(
        pd.cut(
            df["small_fraction"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=[
                "0-25%",
                "25-50%",
                "50-75%",
                "75-100%",
            ],
        ).value_counts().sort_index()
    )


if __name__ == "__main__":
    main()
    