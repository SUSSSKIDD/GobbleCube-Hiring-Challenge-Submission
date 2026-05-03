#!/usr/bin/env python
"""
Improved LightGBM trainer.
Trains on full 37M rows (default) or 1M sample (--sample flag).

Features added vs baseline:
  - Haversine distance from zone centroids
  - Zone-pair historical mean/median (strong lookup signal)
  - Zone-level pickup/dropoff mean (handles sparse pairs)
  - Hour x pickup_zone interaction mean (JFK at 2am vs 5pm)
  - Cyclic time encoding, rush hour, weekend, late night flags
  - Borough-level features
  - Log1p target transform

Usage:
  python train_lgbm.py           # full 37M rows, ~25 min on M2
  python train_lgbm.py --sample  # 1M rows, ~2 min
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
META_PATH = DATA_DIR / "taxi_zone_meta.csv"

AIRPORTS = frozenset({1, 132, 138})  # EWR, JFK, LGA

FEATURE_COLS = [
    "hour", "dow", "month", "week_of_year", "day_of_month",
    "hour_sin", "hour_cos",
    "is_rush_hour", "is_weekend", "is_late_night",
    "passenger_count",
    "pickup_zone", "dropoff_zone",
    "pickup_borough", "dropoff_borough",
    "haversine_km",
    "is_same_zone", "same_borough",
    "is_airport_pickup", "is_airport_dropoff",
    "pair_mean", "pair_median", "log_pair_count",
    "pair_hour_mean",   # zone-pair × hour: 277s pure lookup vs 301s pair-only
    "pair_dow_mean",    # zone-pair × dow
    "pickup_zone_mean", "dropoff_zone_mean",
    "hour_pu_mean", "hour_do_mean",  # hour × dropoff zone (dropoff was #1 importance)
]

# Do NOT mark zones as categorical — default max_cat_threshold=32 means LightGBM
# would only consider 32 of 265 zones per split, ignoring the rest.
# Zones as int32 + histogram binning handles them correctly.
# Borough has -1 sentinel -> NaN in categorical mode, breaking splits.
CAT_FEATURES = []


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_lookup_stats(train: pd.DataFrame):
    global_mean = float(train["duration_seconds"].mean())
    global_median = float(train["duration_seconds"].median())

    pair_agg = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "pair_mean", "median": "pair_median", "count": "pair_count"})
    )

    pu_mean = train.groupby("pickup_zone")["duration_seconds"].mean().rename("pickup_zone_mean")
    do_mean = train.groupby("dropoff_zone")["duration_seconds"].mean().rename("dropoff_zone_mean")

    ts = pd.to_datetime(train["requested_at"])
    hour = ts.dt.hour.astype("int8")
    dow = ts.dt.dayofweek.astype("int8")

    hour_pu_mean = (
        train.assign(_hour=hour)
        .groupby(["_hour", "pickup_zone"])["duration_seconds"].mean()
    )
    hour_pu_mean.index.names = ["hour", "pickup_zone"]

    hour_do_mean = (
        train.assign(_hour=hour)
        .groupby(["_hour", "dropoff_zone"])["duration_seconds"].mean()
    )
    hour_do_mean.index.names = ["hour", "dropoff_zone"]

    pair_hour_mean = (
        train.assign(_hour=hour)
        .groupby(["pickup_zone", "dropoff_zone", "_hour"])["duration_seconds"].mean()
    )
    pair_hour_mean.index.names = ["pickup_zone", "dropoff_zone", "hour"]

    pair_dow_mean = (
        train.assign(_dow=dow)
        .groupby(["pickup_zone", "dropoff_zone", "_dow"])["duration_seconds"].mean()
    )
    pair_dow_mean.index.names = ["pickup_zone", "dropoff_zone", "dow"]

    return {
        "pair_agg": pair_agg,
        "pu_mean": pu_mean.to_dict(),
        "do_mean": do_mean.to_dict(),
        "hour_pu_mean": hour_pu_mean,
        "hour_do_mean": hour_do_mean,
        "pair_hour_mean": pair_hour_mean,
        "pair_dow_mean": pair_dow_mean,
        "global_mean": global_mean,
        "global_median": global_median,
    }


def engineer_features(df: pd.DataFrame, zone_meta: pd.DataFrame, stats: dict) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.astype("int8")
    dow = ts.dt.dayofweek.astype("int8")

    pu = df["pickup_zone"].astype("int32").reset_index(drop=True)
    do = df["dropoff_zone"].astype("int32").reset_index(drop=True)

    # Zone metadata via vectorized map
    meta = zone_meta.set_index("LocationID")
    pu_lat = pu.map(meta["lat"]).fillna(40.75).astype("float32")
    pu_lon = pu.map(meta["lon"]).fillna(-73.98).astype("float32")
    do_lat = do.map(meta["lat"]).fillna(40.75).astype("float32")
    do_lon = do.map(meta["lon"]).fillna(-73.98).astype("float32")

    borough_cats = meta["Borough"].astype("category").cat.codes
    pu_boro = pu.map(borough_cats).fillna(-1).astype("int8")
    do_boro = do.map(borough_cats).fillna(-1).astype("int8")

    dist = haversine_km(pu_lat.values, pu_lon.values, do_lat.values, do_lon.values).astype("float32")

    # All lookups via vectorized merge — no Python loops
    pair_agg = stats["pair_agg"].reset_index()
    pu_mean_s = pd.Series(stats["pu_mean"])
    do_mean_s = pd.Series(stats["do_mean"])
    global_mean = stats["global_mean"]
    global_median = stats["global_median"]

    tmp = pd.DataFrame({
        "pickup_zone": pu.values, "dropoff_zone": do.values,
        "hour": hour.values, "dow": dow.values,
    })
    tmp = tmp.merge(pair_agg, on=["pickup_zone", "dropoff_zone"], how="left")
    pu_fallback = pu.map(pu_mean_s).fillna(global_mean)
    pu_fallback.index = tmp.index
    tmp["pair_mean"] = tmp["pair_mean"].fillna(pu_fallback)
    tmp["pair_median"] = tmp["pair_median"].fillna(global_median)
    tmp["log_pair_count"] = np.log1p(tmp["pair_count"].fillna(0)).astype("float32")

    tmp["pickup_zone_mean"] = pu.map(pu_mean_s).fillna(global_mean).values
    tmp["dropoff_zone_mean"] = do.map(do_mean_s).fillna(global_mean).values

    def _merge_multiindex(tmp, series, join_cols, col_name, fallback_col):
        flat = series.reset_index()
        flat.columns = join_cols + [col_name]
        tmp = tmp.merge(flat, on=join_cols, how="left")
        tmp[col_name] = tmp[col_name].fillna(tmp[fallback_col])
        return tmp

    tmp = _merge_multiindex(tmp, stats["hour_pu_mean"], ["hour", "pickup_zone"],  "hour_pu_mean",  "pickup_zone_mean")
    tmp = _merge_multiindex(tmp, stats["hour_do_mean"], ["hour", "dropoff_zone"], "hour_do_mean",  "dropoff_zone_mean")
    tmp = _merge_multiindex(tmp, stats["pair_hour_mean"], ["pickup_zone", "dropoff_zone", "hour"], "pair_hour_mean", "pair_mean")
    tmp = _merge_multiindex(tmp, stats["pair_dow_mean"],  ["pickup_zone", "dropoff_zone", "dow"],  "pair_dow_mean",  "pair_mean")

    feat = pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "month": ts.dt.month.astype("int8"),
        "week_of_year": ts.dt.isocalendar().week.astype("int16").values,
        "day_of_month": ts.dt.day.astype("int8"),
        "hour_sin": np.sin(2 * np.pi * hour / 24).astype("float32"),
        "hour_cos": np.cos(2 * np.pi * hour / 24).astype("float32"),
        "is_rush_hour": (((hour.between(7, 9)) | (hour.between(16, 19))) & (dow < 5)).astype("int8"),
        "is_weekend": (dow >= 5).astype("int8"),
        "is_late_night": (hour <= 4).astype("int8"),
        "passenger_count": df["passenger_count"].fillna(1).clip(0, 9).astype("int8"),
        "pickup_zone": pu,
        "dropoff_zone": do,
        "pickup_borough": pu_boro,
        "dropoff_borough": do_boro,
        "haversine_km": dist,
        "is_same_zone": (pu == do).astype("int8"),
        "same_borough": (pu_boro == do_boro).astype("int8"),
        "is_airport_pickup": pu.isin(AIRPORTS).astype("int8"),
        "is_airport_dropoff": do.isin(AIRPORTS).astype("int8"),
        "pair_mean": tmp["pair_mean"].astype("float32").values,
        "pair_median": tmp["pair_median"].astype("float32").values,
        "log_pair_count": tmp["log_pair_count"].values,
        "pair_hour_mean": tmp["pair_hour_mean"].astype("float32").values,
        "pair_dow_mean": tmp["pair_dow_mean"].astype("float32").values,
        "pickup_zone_mean": tmp["pickup_zone_mean"].astype("float32").values,
        "dropoff_zone_mean": tmp["dropoff_zone_mean"].astype("float32").values,
        "hour_pu_mean": tmp["hour_pu_mean"].astype("float32").values,
        "hour_do_mean": tmp["hour_do_mean"].astype("float32").values,
    })

    return feat[FEATURE_COLS]


def tune_blend_weight(lgbm_preds, lookup_preds, y_true):
    """Grid search blend weight on dev set."""
    best_w, best_mae = 1.0, np.inf
    for w in np.arange(0.0, 1.01, 0.02):
        blended = w * lgbm_preds + (1 - w) * lookup_preds
        mae = np.mean(np.abs(blended - y_true))
        if mae < best_mae:
            best_mae = mae
            best_w = w
    return best_w, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Train on 1M sample")
    args = parser.parse_args()

    train_file = "sample_1M.parquet" if args.sample else "train.parquet"
    print(f"Loading {train_file}...")
    train = pd.read_parquet(DATA_DIR / train_file)
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    zone_meta = pd.read_csv(META_PATH)
    print(f"  train: {len(train):,}  dev: {len(dev):,}")

    print("Building lookup stats...")
    t0 = time.time()
    stats = build_lookup_stats(train)
    print(f"  done in {time.time()-t0:.0f}s")
    print(f"  pair_agg: {len(stats['pair_agg']):,} pairs")
    print(f"  pair_hour: {len(stats['pair_hour_mean']):,} entries")
    print(f"  pair_dow:  {len(stats['pair_dow_mean']):,} entries")

    print("\nEngineering features...")
    t0 = time.time()
    X_train = engineer_features(train, zone_meta, stats)
    print(f"  train features done in {time.time()-t0:.0f}s")
    t0 = time.time()
    X_dev = engineer_features(dev, zone_meta, stats)
    print(f"  dev features done in {time.time()-t0:.0f}s")
    print(f"  shape: {X_train.shape}  |  nulls: {X_train.isnull().sum().sum()}")

    y_train = train["duration_seconds"].values.astype("float32")
    y_dev = dev["duration_seconds"].values
    print(f"  target mean: {y_train.mean():.1f}s  std: {y_train.std():.1f}s")

    print(f"\nTraining LightGBM on {len(X_train):,} rows...")
    print("  (prints every 100 trees — each ~9s/tree so 100 trees ≈ 15min)")
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        num_leaves=255,         # ~9s/tree × 600 = ~90min on M2
        learning_rate=0.05,
        max_bin=255,
        subsample_for_bin=1_000_000,
        force_row_wise=True,    # faster on M2 ARM
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=50,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()

    _last_print = [t0]

    def _progress_cb(env):
        i = env.iteration + 1
        now = time.time()
        elapsed = now - t0
        # print every ~60s or on tree 1
        if i == 1 or (now - _last_print[0]) >= 60:
            eta = elapsed / i * (env.end_iteration - i)
            trees_per_min = i / max(elapsed / 60, 0.01)
            print(
                f"  tree {i:4d}/{env.end_iteration}"
                f"  elapsed {elapsed/60:.1f}min"
                f"  ETA {eta/60:.1f}min"
                f"  ({trees_per_min:.1f} trees/min)",
                flush=True,
            )
            _last_print[0] = now

    model.fit(
        X_train, y_train,
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.log_evaluation(period=0), _progress_cb],
    )
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print("\nEvaluating...")
    lgbm_preds = model.predict(X_dev)

    dev_pairs = list(zip(dev["pickup_zone"], dev["dropoff_zone"]))
    pair_agg = stats["pair_agg"]
    pu_mean_map = stats["pu_mean"]
    global_mean = stats["global_mean"]
    lookup_preds = np.array([
        pair_agg.loc[p, "pair_mean"] if p in pair_agg.index else pu_mean_map.get(p[0], global_mean)
        for p in dev_pairs
    ], dtype="float32")

    lgbm_mae = np.mean(np.abs(lgbm_preds - y_dev))
    print(f"  LightGBM only MAE: {lgbm_mae:.1f}s")

    best_w, best_mae = tune_blend_weight(lgbm_preds, lookup_preds, y_dev)
    print(f"  Best blend weight:  lgbm={best_w:.2f}, lookup={1-best_w:.2f}")
    print(f"  Blended MAE:        {best_mae:.1f}s")

    # Feature importances (top 10)
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(fi.head(10).to_string())

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "cat_features": CAT_FEATURES,
        "pair_agg": stats["pair_agg"],
        "pu_mean": stats["pu_mean"],
        "do_mean": stats["do_mean"],
        "hour_pu_mean": stats["hour_pu_mean"],
        "hour_do_mean": stats["hour_do_mean"],
        "pair_hour_mean": stats["pair_hour_mean"],
        "pair_dow_mean": stats["pair_dow_mean"],
        "global_mean": stats["global_mean"],
        "global_median": stats["global_median"],
        "zone_meta": zone_meta,
        "blend_weight": best_w,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved bundle to {MODEL_PATH}")
    print(f"Final Dev MAE: {best_mae:.1f}s")


if __name__ == "__main__":
    main()
