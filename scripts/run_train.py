import os
import time

import boto3
import typer

from solar_predictor import storage


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
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        AlgorithmSpecification={
            "TrainingImage": f"{os.environ['IMAGE_REPO_URI']}:latest",
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "python",
                "-m",
                "solar_predictor.models.regressor_tree",
            ],
        },
        InputDataConfig=[
            {
                "ChannelName": "features",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{storage.data_bucket}/{features_prefix}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{storage.data_bucket}/{run_prefix}",
            "CompressionType": "NONE",
        },
        Environment={"DATA_BUCKET": storage.data_bucket},
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": "ml.t3.large",
            "VolumeSizeInGB": 5,
        },
        # EnableManagedSpotTraining=True,
        StoppingCondition={"MaxRuntimeInSeconds": 1800, "MaxWaitTimeInSeconds": 3600},
    )
    print(f"Started {job_name}")


if __name__ == "__main__":
    typer.run(main)
