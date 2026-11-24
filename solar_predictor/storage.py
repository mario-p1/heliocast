import json

import boto3
import pandas as pd

from solar_predictor.utils import config

s3 = boto3.client("s3")


def file_exists(bucket: str, key: str):
    return s3.list_objects_v2(Bucket=bucket, Prefix=key).get("KeyCount", 0) > 0


def get_raw_data_prefix(latitude: str, longitude: str):
    return f"raw/lat={latitude},long={longitude}/"


def get_year_sarah3_file_key(raw_data_prefix: str, year: int):
    return f"{raw_data_prefix}year={year}/sarah3.csv"


def get_features_prefix(latitude: str, longitude: str):
    return f"features/lat={latitude},long={longitude}/"


def get_tree_features_file_key(features_prefix: str):
    return f"{features_prefix}tree_features.csv"


def get_state_file_key(prefix: str):
    return f"{prefix}state.json"


def get_run_prefix(latitude: str, longitude: str):
    return f"runs/lat={latitude},long={longitude}/"


def write_state(prefix: str, last_date: str):
    s3.put_object(
        Bucket=config.DATA_BUCKET,
        Key=get_state_file_key(prefix),
        Body=json.dumps({"last_date": last_date}),
    )


def read_state(prefix: str):
    state_file_key = get_state_file_key(prefix)
    if file_exists(config.DATA_BUCKET, state_file_key):
        state_obj = s3.get_object(Bucket=config.DATA_BUCKET, Key=state_file_key)
        state = json.loads(state_obj["Body"].read())
        print(f"Found state file, last date: {state['last_date']}")
        return state
    return None


def write_csv(file_key: str, df: pd.DataFrame, index: bool = True):
    s3.put_object(
        Bucket=config.DATA_BUCKET,
        Key=file_key,
        Body=df.to_csv(index=index),
    )


def read_s3_csv(file_key: str):
    obj = s3.get_object(Bucket=config.DATA_BUCKET, Key=file_key)
    return pd.read_csv(obj["Body"])


def find_files_at_loc(at_location_prefix: str, filename_end: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(
            Bucket=config.DATA_BUCKET, Prefix=at_location_prefix
        )
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(filename_end)
    ]

    return sorted(keys)
