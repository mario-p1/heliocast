import typer

from solar_predictor.handlers.build_tree_features import lambda_handler


def main(
    lat: str | None = typer.Option(None, help="Latitude"),
    long: str | None = typer.Option(None, help="Longitude"),
    local_path: str | None = typer.Option(None, help="Path to a local raw CSV"),
):
    event = {
        "lat": lat,
        "long": long,
        "local_path": local_path,
    }
    res = lambda_handler(event, {})
    print(res)


if __name__ == "__main__":
    typer.run(main)
