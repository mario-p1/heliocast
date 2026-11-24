import subprocess

import boto3
import mlflow
import typer

from solar_predictor.utils import config


def main(
    arn: str = typer.Argument(..., help="SageMaker model package ARN"),
    image_tag: str = typer.Option("latest", help="Serving image tag"),
):
    sagemaker = boto3.client("sagemaker")
    package = sagemaker.describe_model_package(ModelPackageName=arn)
    if package["ModelApprovalStatus"] != "Approved":
        raise SystemExit(f"{arn} is not approved")

    model_version = mlflow.search_model_versions(
        filter_string=f"tags.`sagemaker.model_package_arn` = '{arn}'"
    )[0]
    artifact_uri = mlflow.MlflowClient().get_model_version_download_uri(
        model_version.name, model_version.version
    )

    subprocess.run(
        (
            "aws cloudformation deploy"
            " --template-file cloudformation-endpoint.yaml"
            " --stack-name radiation-forecaster-endpoint"
            " --parameter-overrides "
            f" ImageUri={config.IMAGE_REPO_URI}:{image_tag}"
            f" ExecutionRoleArn={config.SAGEMAKER_ROLE_ARN}"
            f" ModelDataUri={artifact_uri}/"
        ),
        check=False,
        shell=True,
    )


if __name__ == "__main__":
    typer.run(main)
