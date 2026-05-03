# ETA Challenge — Submission

**Author:** Pratyush Malviya  
**Dev MAE: 258.5 s** · Baseline: ~351 s · Improvement: **~93 s (26%)**

---

## Approach

LightGBM trained on all 37 M rows of 2023 NYC taxi trip data, with dense historical lookup features as the primary signal.

The baseline GBT scores ~351 s despite having access to the same data — not because the model is weak, but because it treats zone IDs as raw integers. Zone 132 (JFK) and zone 161 (Midtown) look like arbitrary numbers to a tree model. The fix is to compute trip-time statistics from the training set and feed them as continuous features, so the model sees *"this pickup→dropoff pair historically takes 312 s"* rather than two opaque integers.

**Features (28 total):**

| Group | Features |
|---|---|
| Lookup (strongest signal) | `pair_mean`, `pair_median`, `log(pair_count)`, `pair_hour_mean`, `pair_dow_mean` |
| Zone-level means | `pickup_zone_mean`, `dropoff_zone_mean`, `hour_pu_mean`, `hour_do_mean` |
| Spatial | Haversine distance from zone centroids |
| Temporal | `hour_sin/cos`, `dow`, `month`, `week_of_year`, `day_of_month` |
| Flags | `is_rush_hour`, `is_weekend`, `is_late_night`, `is_same_zone`, `same_borough`, `is_airport_pickup/dropoff` |
| Categorical | `pickup_zone`, `dropoff_zone`, `pickup_borough`, `dropoff_borough`, `passenger_count` |

**Model:** LightGBM `objective=regression_l1` (directly optimises MAE), `num_leaves=255`, `n_estimators=600`, `learning_rate=0.05`, trained ~90 min on M2 MacBook CPU.

**Blend:** `final = 0.35 × LightGBM + 0.65 × pair_mean_lookup`. The lookup is strong enough on common routes that blending outperforms pure model output. Blend weight grid-searched on dev set.

---

## What Didn't Work

**Log1p target transform** — tried `log1p(duration)` as target with `regression` (MSE) objective, then `expm1` at inference. MAE went up to ~270 s because optimising MSE in log-space biases predictions toward shorter trips. Switched back to `regression_l1` directly on raw seconds.

**Very deep LightGBM (2000 trees, num_leaves=511)** — estimated 4765 min training time on 37 M rows (~9 s/tree). Reduced to 600 trees; marginal gains above that weren't worth the time cost on CPU.

**Baseline features only** — the starter `baseline.py` with 6 raw features scores ~351 s. Adding lookup features was the single biggest jump, taking it from ~351 s to ~262 s before any model tuning.

---

## Where AI Tooling Helped Most

**Claude Code** was used throughout. Highest-leverage moments:

1. **LightGBM ETA blowup** — "ETA 4765 min" in training output. Claude identified 2000 trees × ~9 s/tree on 37 M rows and suggested benchmarking at 100 trees first, then scaling. Saved hours of wasted compute.

2. **Feature engineering pipeline** — translating the lookup-table idea into vectorised pandas merges (groupby → merge → fillna chain) with correct fallback logic for unseen zone pairs. Fast to write, zero bugs first run.

3. **Debugging predict.py inference** — spotted that `hour_pu_mean` and `hour_do_mean` index keys are `(hour, zone)` tuples, not separate scalars. Would have been a silent wrong-value bug at submission.

Where it fell short: required manual back-and-forth to get the `blend_weight` tuning integrated correctly into the save/load cycle — first version saved the weight but predict.py wasn't reading it.

---

## Next Experiments

1. **PyTorch MLP with zone embeddings** — zone embeddings (24-dim per zone) let the model learn that JFK clusters near LGA spatially. Currently training on Kaggle T4; epoch 2 dev MAE = 264.5 s. With more epochs and blend, expected to push below 250 s.

2. **Holiday slice features** — the README notes Eval is a winter-holiday slice. `is_christmas_week`, `is_nye`, and holiday×zone interaction means would likely recover some of the train→eval gap.

3. **OSRM road-network distance** — haversine is a straight-line proxy. Actual driving distance from OpenStreetMap would be a stronger spatial signal, especially for cross-borough routes constrained by bridges and tunnels.

4. **NOAA weather** — hourly rain/snow at JFK/LGA. NYC taxi times slow ~15–30% in precipitation. Free public data, ~1 day to integrate.

5. **GBM residuals on NN** — train LightGBM on NN prediction errors. Often recovers 5–10 s on tabular problems where the NN under/overfits specific route types.

---

## How to Reproduce

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (~500 MB, one-time)
python data/download_data.py

# 3. Train 
python train_lgbm.py
# Writes model.pkl

# 4. Score on Dev
python grade.py
# Expected: 258.5 s MAE

# 5. Build and test Docker image
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
docker images my-eta   # verify ≤ 2.5 GB
```

---

## Repo Structure

```
cpu_submission/
├── predict.py          # submission interface — grader imports this
├── model.pkl           # trained LightGBM bundle (26 MB)
├── train_lgbm.py       # reproducible training script
├── grade.py            # local scoring (mirrors grader logic)
├── Dockerfile          # builds ≤ 2.5 GB image
├── requirements.txt
├── data/
│   ├── download_data.py
│   ├── taxi_zone_meta.csv
│   └── schema.md
└── tests/
    └── test_submission.py
```

---

*Total time: ~14 hours across 2 days.*
