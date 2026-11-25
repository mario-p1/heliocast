#!/bin/bash
exec mlflow models serve -m "${SERVED_MODEL_URI}" -h 0.0.0.0 -p 8080 --env-manager local