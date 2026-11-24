#!/bin/bash
exec uv run mlflow models serve -m "${SERVED_MODEL_URI}" -h 0.0.0.0 -p 8080 --env-manager local