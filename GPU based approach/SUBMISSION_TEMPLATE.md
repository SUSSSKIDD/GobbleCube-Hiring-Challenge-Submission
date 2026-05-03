# Submission Writeup

---

## Final Score

Dev MAE: **255.9 s**  
Baseline (GBT, naive): ~351 s Dev / ~367 s Eval  
**Improvement: ~95 s over baseline (~27%)**

---

## Approach

**4-way ensemble: GPU XGBoost + GPU LightGBM + PyTorch ETANet + lookup blend.**

Trained entirely on Kaggle T4 GPU (free tier). The key insight: zone-pair historical statistics are extremely predictive — the baseline loses because it treats zone IDs as raw integers with no historical signal. Every model in the ensemble gets these lookup features as inputs.

**Features (34 total):**  
- Zone-pair lookups: historical mean/median/std/count, pair×hour, pair×day-of-week, pair×month
- Zone-level: pickup mean, dropoff mean, hour×pickup, hour×dropoff, borough-pair mean
- Spatial: haversine distance (km), Manhattan distance (km)
- Time: hour, dow, month, day, week, hour_sin/cos, dow_norm
- Categorical (as float): pickup_zone, dropoff_zone, pu_boro, do_boro
- Flags: same_zone, same_boro, is_airport_pu, is_airport_do, is_holiday, is_christmas, is_nye
- passenger_count

**XGBoost (GPU):** `tree_method='hist', device='cuda', objective='reg:absoluteerror'`, 1000 rounds (best ~400), dev MAE 262.2 s

**LightGBM (GPU):** `device='gpu', objective='regression_l1'`, 500 rounds, dev MAE 264.4 s

**ETANet (PyTorch):** Zone embeddings (266→16) + boro (8→4) + hour (24→6) + dow (7→4) + 34 continuous + xgb_pred → 256→128→64→1. Trained 6 epochs with OneCycleLR, early stopping. Dev MAE 263.6 s

**Blend weights:** `xgb=0.180, lgb=0.320, nn=0.260, lookup=0.240` → **255.9 s dev MAE**

---

## What Didn't Work

**Fold LGB weight into XGB on CPU** — GPU XGB predictions differ numerically on CPU (different floating-point behavior). Local grade.py showed ~500 s MAE, Kaggle score was correct at 255.9 s. Don't evaluate GPU models on macOS ARM.

**LightGBM 1000 rounds** — Peak at round 200 (261.998 s), rising to 264.353 s at round 1000. Overfitting. Best iteration is ~200 for this dataset.

**Pure NN without lookup features** — Epoch 1 dev MAE ~509 s. Adding pair_mean lookup as input feature → epoch 1 dropped to reasonable range. Lookup signal is load-bearing.

**Log1p target** — MAE in log-space biases toward short trips. Raw-target with `reg:absoluteerror` / `regression_l1` wins.

---

## Where AI Tooling Sped Up Most

**Claude Code** used throughout. Highest-leverage moments:

1. **GPU platform incompatibility diagnosis** — LightGBM GPU model segfaults on macOS ARM. Claude identified `[device_type: gpu]` header in `.txt` file as the cause and designed the slim-bundle approach (separate `.ubj`/`.txt` files, no xgb/lgb objects in pickle).

2. **ETANet from scratch** — Zone/boro/hour/dow embeddings + continuous features + xgb_pred stacking input, full DataLoader + OneCycleLR + per-100-batch ETA progress. Would have taken hours manually.

3. **Kernel restart recovery** — After CUDA arch mismatch killed the kernel, Claude wrote a recovery cell (Cell 7b) that rebuilt all features and model predictions from disk files without retraining.

4. **NaN guard for zones 264/265** — Zone centroids have NaN lat/lon in the metadata. Claude caught this and added the `_get_latlon()` helper.

---

## How to Reproduce

**On Kaggle (T4 GPU, recommended):**
```bash
# 1. Upload colab/train_kaggle_v2.ipynb to Kaggle with T4 GPU
# 2. Run all cells (~3–4 hours total)
#    Cell 6: XGBoost GPU (~30 min, 1000 rounds)
#    Cell 7: LightGBM GPU (~20 min, 500 rounds)
#    Cell 8: ETANet (~1.5 hr, 6 epochs early stopping)
#    Cell 9: Blend + eval → 255.9 s
#    Cell 11b: Save model_ensemble_slim.pkl
# 3. Download: xgb_model.ubj, lgb_model.txt, model_ensemble_slim.pkl (rename to model.pkl)
```

**Local Docker (CPU inference, Linux only):**
```bash
cd gpu_submission
docker build -t my-eta-gpu .
docker run --rm -v $(pwd)/data:/work my-eta-gpu /work/dev.parquet /work/preds.csv
```

Note: LightGBM `.txt` file was saved with `[device_type: gpu]` — requires `lightgbm>=4.0` built with GPU support, or Linux with OpenCL drivers. macOS ARM cannot load this model.

---

*Total time: ~18 hours across 3 days.*
