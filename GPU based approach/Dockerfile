# GPU Ensemble Submission — ETA Challenge
# Dev MAE: 255.9 s (XGBoost 0.18 + LightGBM 0.32 + NN 0.26 + lookup 0.24)
#
# Build:
#   docker build -t my-eta-gpu .
# Test:
#   docker run --rm -v $(pwd)/data:/work my-eta-gpu /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py ./
COPY model.pkl ./
COPY xgb_model.ubj ./
COPY lgb_model.txt ./

ENTRYPOINT ["python", "grade.py"]
