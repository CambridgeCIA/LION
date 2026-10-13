mlflow server \
  --host 127.0.0.1 --port 5000 \
  --serve-artifacts \
  --backend-store-uri sqlite:////store/LION/ttv22/pcm/results/mlflow/mlflow.db \
  --artifacts-destination /store/LION/ttv22/pcm/results/mlflow
#   --default-artifact-root mlflow-artifacts:/
