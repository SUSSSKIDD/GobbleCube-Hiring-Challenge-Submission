# GobbleCube ETA Challenge — Submission

NYC taxi trip duration prediction. Two approaches submitted, progressively improving from baseline.

| Approach | Dev MAE | vs Baseline |
|----------|---------|-------------|
| Baseline (provided GBT) | ~351 s | — |
| CPU: LightGBM + lookup blend | 258.5 s | −93 s (−25%) |
| **GPU: XGBoost + LightGBM + ETANet + lookup** | **255.9 s** | **−95 s (−27%)** |

---

## Repository Structure

```
GobbleCube AI submission/
├── CPU based approach/      # LightGBM trained on macOS CPU
│   ├── predict.py
│   ├── grade.py
│   ├── train_lgbm.py
│   ├── model.pkl
│   ├── Dockerfile
│   └── requirements.txt
│
├── GPU based approach/      # 4-model ensemble trained on Kaggle T4 GPU
│   ├── predict.py
│   ├── grade.py
│   ├── model.pkl            # slim bundle (lookup stats + NN weights)
│   ├── xgb_model.ubj        # XGBoost binary
│   ├── lgb_model.txt        # LightGBM text
│   ├── Dockerfile
│   └── requirements.txt
│
└── README.md                # this file
```

---

## Approach 1 — CPU: LightGBM + Lookup Blend

**Dev MAE: 258.5 s**

### What it does

LightGBM (`objective=regression_l1`, 600 trees, `num_leaves=255`) trained on all 37 M rows of 2023 NYC trip data. The key insight is that raw zone IDs carry almost no signal — the model needs historical trip-time statistics as features.

**28 features:** zone-pair historical mean/median/count, pair×hour and pair×dow interaction means, pickup/dropoff zone means, hour×zone means, haversine distance from zone centroids, cyclic time encodings (hour sin/cos, dow, month, week), airport flags, same-zone/same-borough flags, passenger count.

**Final prediction:** `0.35 × LightGBM + 0.65 × pair_mean_lookup`  
The lookup alone is surprisingly strong on common routes — blending beats the pure model.

### Run

```bash
cd "CPU based approach"
pip install -r requirements.txt

# retrain (optional, ~90 min on M2 MacBook)
python train_lgbm.py

# score
python grade.py data/dev.parquet preds.csv
# Expected: ~258.5 s

# Docker
docker build -t my-eta-cpu .
docker run --rm -v $(pwd)/data:/work my-eta-cpu /work/dev.parquet /work/preds.csv
```

---

## Approach 2 — GPU: 4-Model Ensemble

**Dev MAE: 255.9 s**

### What it does

Four models trained on Kaggle T4 GPU, blended by optimised weights:

| Model | Dev MAE | Weight |
|-------|---------|--------|
| XGBoost GPU (`reg:absoluteerror`, 1000 rounds) | 262.2 s | 0.18 |
| LightGBM GPU (`regression_l1`, 500 rounds) | 264.4 s | 0.32 |
| ETANet — PyTorch MLP with zone embeddings | 263.6 s | 0.26 |
| Zone-pair lookup mean | — | 0.24 |
| **Ensemble** | **255.9 s** | — |

All models share the same 34 engineered features (superset of the CPU approach, adding month interactions, boro-pair means, and richer spatial features).

**ETANet architecture:** Zone embeddings (266→16) + borough (8→4) + hour (24→6) + dow (7→4) + 34 continuous features + XGBoost prediction as stacking input → 256 → 128 → 64 → 1. Trained with OneCycleLR, early stopping at epoch 6.

### Download Model Weights

Weights are hosted on Google Drive (too large for git):

**[Download from Google Drive](https://drive.google.com/drive/folders/1OokbmLg7jcxBCWTtbovCp6f8sUW5SdI8?usp=sharing)**

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1OokbmLg7jcxBCWTtbovCp6f8sUW5SdI8
# place model.pkl, xgb_model.ubj, lgb_model.txt into "GPU based approach/"
```

### Run — Kaggle (no local GPU needed)

1. New notebook on [kaggle.com/code](https://www.kaggle.com/code) → upload model files + `predict.py` as a dataset
2. Run:

```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, '/kaggle/input/<your-dataset-name>')
import predict

dev = pd.read_parquet('/kaggle/input/gobblecube-eta-challenge/dev.parquet')
preds = dev.apply(lambda r: predict.predict({
    'pickup_zone':     r['pickup_zone'],
    'dropoff_zone':    r['dropoff_zone'],
    'passenger_count': r.get('passenger_count', 1),
    'requested_at':    str(r['requested_at']),
}), axis=1)

print(f"Dev MAE: {np.mean(np.abs(preds.values - dev['duration'].values)):.1f} s")
# Expected: 255.9 s
```

### Run — Local GPU (Linux + NVIDIA)

```bash
cd "GPU based approach"
pip install "numpy<2" pandas torch xgboost>=2.0 lightgbm>=4.0

python grade.py data/dev.parquet preds.csv
# Expected: ~255.9 s

# Docker
docker build -t my-eta-gpu .
docker run --rm --gpus all \
    -v $(pwd)/data:/work \
    my-eta-gpu /work/dev.parquet /work/preds.csv
```

### Run — macOS CPU

```bash
cd "GPU based approach"
pip install "numpy<2" pandas torch xgboost>=2.0 lightgbm>=4.0
python grade.py data/dev.parquet preds.csv
```

---

## What Didn't Work

**Log1p target transform** — optimising MSE in log-space biases toward short trips. Raw target with `reg:absoluteerror` / `regression_l1` wins.

**LightGBM 1000 rounds** — peak at round 200 (262 s), rising to 264 s at round 1000. Overfitting. Capped at 500.

**Very deep LightGBM (2000 trees, `num_leaves=511`)** — estimated 4765 min training on 37 M rows. Reduced to 600 trees; gains plateau before that.

**Pure NN without lookup features** — epoch-1 dev MAE ~509 s. Adding `pair_mean` as a continuous input dropped it immediately. The lookup signal is load-bearing.

---

## Where Claude Code Helped Most

1. **Feature engineering pipeline** — zone-pair stats, hour/dow/month interactions, haversine with NaN guard for missing zone centroids. Built and debugged in one session.

2. **ETANet from scratch** — embeddings + continuous features + XGBoost stacking input, full DataLoader + OneCycleLR + per-100-batch ETA progress display.

3. **Kaggle kernel crash recovery** — after CUDA arch mismatch killed the kernel mid-training, wrote a recovery cell that reloaded all models and features from disk without retraining.

4. **Cross-platform model loading** — diagnosed segfaults from pickling GPU XGBoost/LightGBM on macOS ARM; designed the slim-bundle pattern (`.ubj` / `.txt` separate from `model.pkl`) and patched the `device_type: gpu` header in `lgb_model.txt`.

5. **NaN predictions** — `arcsin(sqrt(a))` where `a` slightly exceeded 1.0 due to float precision. Fixed with `np.clip(a, 0.0, 1.0)`.

---

## Reproduce GPU Training from Scratch

Requires Kaggle T4 GPU (~3–4 hours total):

| Cell | Task | Time | Dev MAE |
|------|------|------|---------|
| 1–5 | Load data, engineer 34 features | ~10 min | — |
| 6 | XGBoost GPU, 1000 rounds | ~30 min | 262.2 s |
| 7 | LightGBM GPU, 500 rounds | ~20 min | 264.4 s |
| 8 | ETANet, early stop @ epoch 6 | ~1.5 hr | 263.6 s |
| 9 | Blend + evaluate | ~2 min | **255.9 s** |
| 11b | Export `model_ensemble_slim.pkl` | instant | — |

Upload `colab/train_kaggle_v2.ipynb` to Kaggle with T4 GPU enabled. If the kernel restarts mid-run, use Cell 7b to restore all state from disk without retraining.

---

*Total time: ~18 hours across 3 days.*
