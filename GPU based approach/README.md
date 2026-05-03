# ETA Challenge — GPU Ensemble Submission

**Dev MAE: 255.9 s** — XGBoost (GPU) + LightGBM (GPU) + ETANet (PyTorch) + lookup blend  
Baseline: ~351 s &nbsp;|&nbsp; Improvement: **~95 s (~27%)**

---

## Overview

A 4-model ensemble trained entirely on Kaggle T4 GPU:

| Model | Dev MAE | Blend Weight |
|-------|---------|-------------|
| XGBoost (GPU, 1000 rounds) | 262.2 s | 0.18 |
| LightGBM (GPU, 500 rounds) | 264.4 s | 0.32 |
| ETANet (PyTorch, 6 epochs) | 263.6 s | 0.26 |
| Zone-pair lookup mean | — | 0.24 |
| **Ensemble** | **255.9 s** | — |

All models share the same 34 engineered features: zone-pair historical statistics, hour/dow/month interactions, haversine distance, cyclic time encodings, airport flags, and passenger count.

---

## Step 1 — Download Model Weights

Weights are hosted on Google Drive (too large for git):

**[Download from Google Drive](https://drive.google.com/drive/folders/1OokbmLg7jcxBCWTtbovCp6f8sUW5SdI8?usp=sharing)**

Or pull via terminal:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1OokbmLg7jcxBCWTtbovCp6f8sUW5SdI8
```

Place these three files in this directory alongside `predict.py`:

| File | Size | Contents |
|------|------|----------|
| `model.pkl` | ~16 MB | Lookup stats + NN weights (no xgb/lgb objects) |
| `xgb_model.ubj` | ~4.7 MB | XGBoost model, binary UBJ format |
| `lgb_model.txt` | ~23 MB | LightGBM model, text format |

---

## Step 2 — Run Inference

### Option A — Kaggle Notebook (no local GPU needed)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Upload `model.pkl`, `xgb_model.ubj`, `lgb_model.txt`, `predict.py` as a dataset
3. Run:

```python
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/kaggle/input/<your-dataset-name>')
import predict

dev = pd.read_parquet('/kaggle/input/gobblecube-eta-challenge/dev.parquet')

preds = dev.apply(lambda r: predict.predict({
    'pickup_zone':     r['pickup_zone'],
    'dropoff_zone':    r['dropoff_zone'],
    'passenger_count': r.get('passenger_count', 1),
    'requested_at':    str(r['requested_at']),
}), axis=1)

mae = np.mean(np.abs(preds.values - dev['duration'].values))
print(f'Dev MAE: {mae:.1f} s')  # expected: 255.9 s

preds.to_csv('preds.csv', index=False)
```

---

### Option B — Local GPU (Linux + NVIDIA)

**Requirements:** CUDA 11.8+, Docker with `nvidia-docker2` or `--gpus` support.

**Install and score directly:**

```bash
pip install "numpy<2" pandas torch xgboost>=2.0 lightgbm>=4.0

# place dev.parquet in data/
python grade.py data/dev.parquet preds.csv
# Expected: Dev MAE ≈ 255.9 s
```

**Or via Docker:**

```bash
docker build -t my-eta-gpu .
docker run --rm --gpus all \
    -v $(pwd)/data:/work \
    my-eta-gpu /work/dev.parquet /work/preds.csv
```

> Inference itself runs on CPU (`torch.no_grad()`, XGBoost/LightGBM CPU predict). `--gpus all` is optional.

---

### Option C — macOS (CPU only)

Works without Docker. The LightGBM model header has been patched from `device_type: gpu` to `device_type: cpu` — weights are unchanged.

```bash
pip install "numpy<2" pandas torch xgboost>=2.0 lightgbm>=4.0

# smoke test
python -c "
from predict import predict
print(predict({
    'pickup_zone': 161,
    'dropoff_zone': 236,
    'passenger_count': 1,
    'requested_at': '2023-12-15T08:30:00'
}))
# Expected: ~514 s
"

# score on dev set
python grade.py data/dev.parquet preds.csv
```

---

## Reproduce Training from Scratch

Training requires a **Kaggle T4 GPU** notebook (~3–4 hours total).

1. Upload `colab/train_kaggle_v2.ipynb` to [kaggle.com/code](https://www.kaggle.com/code) with **T4 GPU** enabled
2. Run cells in order:

| Cell | Task | Duration | Dev MAE |
|------|------|----------|---------|
| 1–5 | Load data, engineer features | ~10 min | — |
| 6 | Train XGBoost GPU (1000 rounds) | ~30 min | 262.2 s |
| 7 | Train LightGBM GPU (500 rounds) | ~20 min | 264.4 s |
| 8 | Train ETANet (early stop @ epoch 6) | ~1.5 hr | 263.6 s |
| 9 | Blend + evaluate | ~2 min | **255.9 s** |
| 11b | Export `model_ensemble_slim.pkl` | instant | — |

3. Download `xgb_model.ubj`, `lgb_model.txt`, `model_ensemble_slim.pkl`
4. Rename `model_ensemble_slim.pkl` → `model.pkl`

> **Kernel restart mid-run?** Run Cell 7b to reload all models and features from disk, then resume from Cell 8 — no retraining needed.

---

## Project Structure

```
gpu_submission/
├── predict.py            # submission interface — predict(request) -> float
├── grade.py              # grader: reads parquet, calls predict, reports MAE
├── Dockerfile            # python:3.11-slim + libgomp1
├── requirements.txt
├── model.pkl             # download from Drive (not in git)
├── xgb_model.ubj         # download from Drive (not in git)
├── lgb_model.txt         # download from Drive (not in git)
├── data/
│   └── taxi_zone_meta.csv
└── tests/
```

---

## Notes

- `lgb_model.txt` was GPU-trained but patched to `device_type: cpu` — safe to load on any platform.
- Never pickle `xgb.Booster` or `lgb.Booster` objects directly — they segfault on macOS ARM and cross-platform loads. Always use `.ubj` / `.txt` files.
- `model.pkl` is the **slim** bundle — it contains only lookup dicts and NN state dict, no xgb/lgb objects.
