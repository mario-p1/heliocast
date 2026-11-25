# Solar Radiation Predictor

Solar Radiation Predictor is an end-to-end ML project on AWS that predicts hourly shortwave solar radiation for a given location from EUMETSAT SARAH-3 satellite data.
It covers data ingestion, feature building, XGBoost training on spot instances, MLflow experiment tracking and model registry, and serverless deployment on SageMaker.

## AWS architecture
- **CloudFormation**: main stack and a separate endpoint stack
- **S3**: raw data, features, training outputs and MLflow artifacts
- **Lambda**: fetches SARAH-3 radiation data from Open-Meteo into S3
- **ECR**: shared Docker image for preprocessing, training and serving
- **SageMaker Processing**: builds the features
- **SageMaker Training**: trains XGBoost on spot instances
- **SageMaker MLflow**: experiment tracking and model registry
- **SageMaker Serverless Endpoint**: serves a registered MLflow model
- **SageMaker Domain**: Studio domain and user profile
- **IAM**: Lambda and SageMaker execution roles

## ML workflow
1. **Fetch**: the Lambda saves raw radiation data to S3
2. **Preprocess**: `scripts/run_preprocessing.py` starts a SageMaker Processing job that builds the features
3. **Train**: `scripts/run_train.py` starts a SageMaker Training job that logs the run and model to MLflow
4. **Register**: the chosen model is registered manually from the MLflow UI
5. **Deploy**: `scripts/deploy_model.py` deploys the registered model (by version or alias) to the endpoint
6. **Predict**: `scripts/invoke_endpoint.py` sends a sample payload to the endpoint

## Useful commands
```bash
# Invoke local model endpoint with sample payload
curl http://127.0.0.1:8080\/invocations -H "Content-Type:application/json" --data-binary "@sample_payload.json"
```