import time

import boto3
import typer

from heliocast import storage
from heliocast.utils import config


def main(
    lat: str = typer.Option(..., help="Latitude"),
    long: str = typer.Option(..., help="Longitude"),
):
    features_prefix = storage.get_features_prefix(lat, long)
    run_prefix = storage.get_run_prefix(lat, long)

    job_name = f"train-xgboost-{int(time.time())}"

    sagemaker = boto3.client("sagemaker")
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=config.SAGEMAKER_ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": f"{config.IMAGE_REPO_URI}:latest",
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "python",
                "-m",
                "heliocast.models.regressor_tree",
            ],
        },
        InputDataConfig=[
            {
                "ChannelName": "features",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{config.DATA_BUCKET}/{features_prefix}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{config.DATA_BUCKET}/{run_prefix}",
            "CompressionType": "NONE",
        },
        Environment={
            "DATA_BUCKET": config.DATA_BUCKET,
            "MLFLOW_TRACKING_URI": config.MLFLOW_TRACKING_URI,
        },
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": "ml.t3.large",
            "VolumeSizeInGB": 5,
        },
        EnableManagedSpotTraining=True,
        StoppingCondition={"MaxRuntimeInSeconds": 1800, "MaxWaitTimeInSeconds": 3600},
    )
    print(f"Started {job_name}")


if __name__ == "__main__":
    typer.run(main)
