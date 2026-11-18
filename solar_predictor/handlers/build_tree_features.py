from typing import Any

import pandas as pd

from solar_predictor import storage
from solar_predictor.data import drop_incomplete_tail, read_sarah3
from solar_predictor.features import build_tree_features


def lambda_handler(event: dict, context: Any):
    local_path = event.get("local_path")

    if local_path:
        df = pd.read_csv(local_path, parse_dates=["datetime"]).set_index("datetime")
    else:
        df = read_sarah3(storage.get_location_prefix(event["lat"], event["long"]))

    features_df = build_tree_features(drop_incomplete_tail(df))

    if local_path:
        features_df.to_csv("tree_features.csv")
        return {"statusCode": 200, "result": {"rows": len(features_df)}}

    file_key = f"features/lat={event['lat']},long={event['long']}/tree_features.csv"
    storage.write_csv(file_key, features_df)

    return {
        "statusCode": 200,
        "result": {"file_key": file_key, "rows": len(features_df)},
    }
