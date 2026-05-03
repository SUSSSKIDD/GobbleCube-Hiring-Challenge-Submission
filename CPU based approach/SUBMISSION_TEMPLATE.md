# Submission Writeup

---

## Final Score

Dev MAE: **258.5 s**  
Baseline (GBT, naive): ~351 s Dev / ~367 s Eval  
**Improvement: ~93 s over baseline (~25%)**

> A PyTorch MLP with zone embeddings is also in training on Kaggle T4 (epoch 2 dev=264.5 s, converging).
> If it beats 258.5 s before submission, `model.pkl` will be swapped to the NN bundle.

---

## Approach

**LightGBM with dense lookup features** trained on all 37 M rows of 2023 trip data.

The key insight: the baseline GBT loses to a 10-line zone-pair average because it treats zone IDs as raw integers and ignores historical trip-time distributions entirely. The fix is to compute lookup statistics from the training set and feed them as features — so the model sees "this pickup→dropoff pair historically takes 312 s" rather than "pickup_zone=132, dropoff_zone=161".

**Features (28 total):** zone-pair historical mean/median/count, pair×hour mean, pair×day-of-week mean, zone-level pickup/dropoff means, hour×zone interaction means, haversine distance from zone centroids, cyclic time encodings (hour sin/cos, dow, month, week, day), rush hour/weekend/late-night flags, airport flags, same-zone/same-borough flags, passenger count.

**Model:** LightGBM `objective=regression_l1` (directly optimises MAE), `num_leaves=255`, `n_estimators=600`, `learning_rate=0.05`, trained ~90 min on M2 MacBook CPU.

**Blend:** final = `0.35 × LightGBM + 0.65 × pair_mean_lookup`. The lookup is strong enough on common routes that blending outperforms pure model output.

---

## What Didn't Work

**Baseline LightGBM on 6 raw features** — scores ~351 s. Zone IDs as integers carry almost no geographic signal.

**Log1p target transform** — tried `log1p(duration)` + `regression` objective, then `expm1` at inference. MAE went up (~270 s) because optimising MSE in log-space biases predictions toward shorter trips.

**Very deep LightGBM (2000 trees, num_leaves=511)** — estimated 4765 min training time on 37 M rows. Benchmarked at ~9 s/tree. Reduced to 600 trees; marginal gains above that threshold weren't worth the time cost.

**Pure NN without lookup features** — first MLP had epoch-1 dev MAE ~310 s. Adding `pair_mean` and related lookups as continuous inputs dropped epoch-1 dev to ~275 s immediately.

---

## Where AI Tooling Sped Up Most

**Claude Code** was used throughout. Highest-leverage moments:

1. **LightGBM ETA blowup** — pasted "ETA 4765 min" output, Claude identified 2000 trees × ~9 s/tree × 37 M rows and suggested 600 trees. Saved ~3 hrs of wasted compute.

2. **PyTorch MLP from scratch** — zone/borough/hour/dow embeddings + continuous features, full DataLoader + OneCycleLR + checkpoint-resume loop. Would have taken 3–4 hrs manually; ~20 min with Claude.

3. **Colab 12 GB RAM debugging** — memmap chunking strategy (2 M rows/chunk) and `num_workers=0` fix (DataLoader workers fork the process, duplicating 3 GB arrays → OOM) both diagnosed and patched in a single conversation turn.

4. **NaN loss at batch 500** — `running_mae=NaN` traced to float precision in `arcsin(sqrt(a))` where `a` slightly exceeded 1.0. Fix: `np.clip(a, 0.0, 1.0)` + `np.nan_to_num`. One-shot diagnosis.

Where it fell short: Kaggle's broken-torch environment required 3 iterations to fix (`torch._utils` AttributeError). First two Claude fixes didn't fully resolve it — the working solution (force-reinstall `torch==2.2.0 cu118` then restart kernel) needed manual trial.

---

## Next Experiments

1. **More NN epochs** — current run shows dev MAE still falling at epoch 2 (264.5 s). 10–12 epochs likely pushes below 250 s.

2. **Holiday slice features** — Eval is a winter-holiday slice (README note). Adding `is_christmas_week`, `is_nye` flags and holiday×zone interaction means would likely recover some train→eval gap.

3. **OSRM road-network distance** — haversine is a proxy; actual driving distance from OpenStreetMap would be stronger, especially for cross-borough routes where road topology matters.

4. **GBM residuals on NN** — train LightGBM on NN prediction errors. Often recovers 5–10 s on tabular problems where the NN under/overfits specific route types.

5. **NOAA weather** — hourly rain/snow data for JFK/LGA/NYC. Rain causes ~15–30% NYC slowdowns. Free data, ~1 day to integrate.

---

## How to Reproduce

```bash
# 1. Clone and install
git clone <repo-url>
cd eta-challenge-starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (~500 MB, one-time)
python data/download_data.py

# 3. Train LightGBM (~90 min on M2 MacBook, ~25 min on modern desktop)
python train_lgbm.py
# Writes model.pkl

# 4. Score on Dev
python grade.py
# Expected: ~258.5 s MAE

# 5. Build and test Docker image
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
docker images my-eta   # verify ≤ 2.5 GB
```

**NN path (optional):**
```bash
# Upload colab/train_kaggle.ipynb to Kaggle with T4 GPU → run all cells (~2.5 hrs)
# Download model_nn.pkl → copy to eta-challenge-starter/
cp model_nn.pkl model.pkl
cp ../colab/predict_nn.py predict.py
python grade.py   # compare to LightGBM score; keep whichever is lower
```

---

*Total time: ~14 hours across 2 days.*
