import json
import logging
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer
import xgboost as xgb

from heliocast.data import split_train_valid_test_by_index
from heliocast.metrics import calculate_metrics

INPUT_DIR = Path("/opt/ml/input/data/features")
OUTPUT_DIR = Path("/opt/ml/output/data")
MODEL_DIR = Path("/opt/ml/model")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(
    input_dir: Annotated[Path, typer.Option()] = INPUT_DIR,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
    model_dir: Annotated[Path, typer.Option()] = MODEL_DIR,
    train_size: Annotated[float, typer.Option()] = 0.9,
    valid_size: Annotated[float, typer.Option()] = 0.05,
):
    logger.info("Starting training of XGBoost regression model...")

    mlflow.set_experiment("SARAH3-predictor")

    features_df = pd.read_csv(
        input_dir / "tree_features.csv", index_col="datetime", parse_dates=True
    )

    logger.info(f"Loaded {len(features_df)} rows from {input_dir}")

    train_df, valid_df, test_df = split_train_valid_test_by_index(
        features_df, train_size, valid_size
    )

    target = "target"
    X_train, y_train = train_df.drop(columns=target), train_df[target]
    X_valid, y_valid = valid_df.drop(columns=target), valid_df[target]
    X_test, y_test = test_df.drop(columns=target), test_df[target]

    logger.info("Finding best iteration for XGBoost model...")
    model_find_iteration = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
    model_find_iteration.fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100
    )
    logger.info(f"Best iteration: {model_find_iteration.best_iteration}")

    logger.info("Training final XGBoost model...")
    mlflow.xgboost.autolog()
    final_model = xgb.XGBRegressor(n_estimators=model_find_iteration.best_iteration)
    with mlflow.start_run() as run:
        final_model.fit(X_train, y_train)

        logger.info("Evaluating final model...")
        test_metrics = calculate_metrics(y_test.to_numpy(), final_model.predict(X_test))
        mlflow.models.evaluate(
            model=f"runs:/{run.info.run_id}/model",
            data=valid_df,
            targets="target",
            model_type="regressor",
        )
        logger.info(f"Test metrics: {test_metrics}")

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

    final_model.save_model(model_dir / "model.json")
    logger.info(f"Saved model to {model_dir / 'model.json'}")


if __name__ == "__main__":
    typer.run(main)
