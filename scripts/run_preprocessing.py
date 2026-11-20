import os
import time

import boto3
import typer

from solar_predictor import storage


def main(
    lat: str = typer.Option(..., help="Latitude"),
    long: str = typer.Option(..., help="Longitude"),
):
    bucket = storage.data_bucket
    location_prefix = storage.get_location_prefix(lat, long)
    features_prefix = os.path.dirname(storage.get_tree_features_file_key(lat, long))
    job_name = f"build-tree-features-{int(time.time())}"

    sagemaker = boto3.client("sagemaker")
    sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        AppSpecification={
            "ImageUri": f"{os.environ['IMAGE_REPO_URI']}:latest",
            "ContainerEntrypoint": ["python", "-m", "solar_predictor.preprocess"],
        },
        Environment={"DATA_BUCKET": bucket},
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.medium",
                "VolumeSizeInGB": 5,
            }
        },
        ProcessingInputs=[
            {
                "InputName": "raw",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{location_prefix}",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "features",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{features_prefix}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )
    print(f"Started {job_name}")


if __name__ == "__main__":
    typer.run(main)
