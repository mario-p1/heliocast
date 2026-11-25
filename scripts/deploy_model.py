import logging
import subprocess

import boto3
import mlflow
import typer

from solar_predictor.utils import config, setup_logging

logger = logging.getLogger(__name__)


def main(
    model_uri: str = typer.Argument(
        ...,
        help="Model version or alias (e.g. SARAH3-predictor/7 or SARAH3-predictor@champion)",
    ),
    image_tag: str = typer.Option("latest", help="Serving image tag"),
):
    setup_logging()
    logger.info(f"Looking up model {model_uri}")
    client = mlflow.MlflowClient()
    if "@" in model_uri:
        name, alias = model_uri.split("@")
        model_version = client.get_model_version_by_alias(name, alias)
    else:
        name, version = model_uri.split("/")
        model_version = client.get_model_version(name, version)

    image = boto3.client("ecr").describe_images(
        repositoryName=config.IMAGE_REPO_URI.split("/", 1)[1],
        imageIds=[{"imageTag": image_tag}],
    )["imageDetails"][0]

    logger.info(
        f"Deploying {model_version.name} version {model_version.version}"
        f" with image {image_tag} ({image['imageDigest']})"
    )

    subprocess.run(
        (
            "aws cloudformation deploy"
            " --template-file cloudformation-endpoint.yaml"
            " --stack-name radiation-forecaster-endpoint"
            " --parameter-overrides "
            f" ImageUri={config.IMAGE_REPO_URI}@{image['imageDigest']}"
            f" ExecutionRoleArn={config.SAGEMAKER_ROLE_ARN}"
            f" MlflowTrackingUri={config.MLFLOW_TRACKING_URI}"
            f" ModelUri=models:/{model_version.name}/{model_version.version}"
        ),
        check=False,
        shell=True,
    )


if __name__ == "__main__":
    typer.run(main)
