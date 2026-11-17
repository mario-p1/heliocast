from pathlib import Path

import pandas as pd

from solar_predictor.features import build_tree_features


def main():
    df = pd.read_csv(
        Path(__file__).parents[2] / "notebooks" / "raw_df.csv", parse_dates=["datetime"]
    ).set_index("datetime")

    features_df = build_tree_features(df)
    print(features_df.head())
    breakpoint()


if __name__ == "__main__":
    main()
