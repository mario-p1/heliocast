import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from heliocast.data import split_train_valid_test_by_index
from heliocast.metrics import calculate_metrics

INPUT_DIR = Path("/opt/ml/input/data/features")
OUTPUT_DIR = Path("/opt/ml/output/data")
MODEL_DIR = Path("/opt/ml/model")


def main(
    input_dir: Annotated[Path, typer.Option()] = INPUT_DIR,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
    model_dir: Annotated[Path, typer.Option()] = MODEL_DIR,
    train_size: Annotated[float, typer.Option()] = 0.9,
    valid_size: Annotated[float, typer.Option()] = 0.05,
):
    features_df = pd.read_csv(
        input_dir / "tree_features.csv", index_col="datetime", parse_dates=True
    )
    print(f"Loaded {len(features_df)} rows from {input_dir}")

    train_df, valid_df, test_df = split_train_valid_test_by_index(
        features_df, train_size, valid_size
    )

    target = "target"
    X_test, y_test = test_df.drop(columns=target), test_df[target]

    test_metrics = calculate_metrics(y_test.to_numpy(), X_test["lag_49"])

    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value}")

    with open(output_dir / "info.json", "w") as f:
        json.dump(
            {
                "train_period": f"{train_df.index[0]} to {train_df.index[-1]}",
                "valid_period": f"{valid_df.index[0]} to {valid_df.index[-1]}",
                "test_period": f"{test_df.index[0]} to {test_df.index[-1]}",
                "test_metrics": test_metrics,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    typer.run(main)
