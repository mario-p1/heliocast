from pathlib import Path
from typing import Annotated

import boto3
import typer


def main(
    endpoint_name: Annotated[str, typer.Argument(help="SageMaker endpoint name")],
    payload: Annotated[Path, typer.Argument(help="JSON payload file")] = Path(
        "sample_payload.json"
    ),
):
    response = boto3.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload.read_bytes(),
    )
    print(response["Body"].read().decode())


if __name__ == "__main__":
    typer.run(main)
