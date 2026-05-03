"""Submission interface — Gobblecube grader imports this."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_BUNDLE_PATH = Path(__file__).parent / "model.pkl"

with open(_BUNDLE_PATH, "rb") as _f:
    _BUNDLE = pickle.load(_f)

_MODEL = _BUNDLE["model"]
_PU_MEAN = _BUNDLE["pu_mean"]
_DO_MEAN = _BUNDLE["do_mean"]
_GLOBAL_MEAN = _BUNDLE["global_mean"]
_GLOBAL_MEDIAN = _BUNDLE["global_median"]
_BLEND_W = _BUNDLE["blend_weight"]
_FEATURE_COLS = _BUNDLE["feature_cols"]
_META = _BUNDLE["zone_meta"].set_index("LocationID")
_BOROUGH_CATS = _META["Borough"].astype("category").cat.codes
_AIRPORTS = frozenset({1, 132, 138})

def _s2d(series):
    return series.to_dict() if hasattr(series, "to_dict") else series

# Pair lookup
_PAIR_DICT = {
    (int(r["pickup_zone"]), int(r["dropoff_zone"])): (r["pair_mean"], r["pair_median"], r["pair_count"])
    for _, r in _BUNDLE["pair_agg"].reset_index().iterrows()
}

# Optional lookup tables — present in full-train bundle, absent in sample bundle
_HOUR_PU_MEAN  = _s2d(_BUNDLE.get("hour_pu_mean", {}))
_HOUR_DO_MEAN  = _s2d(_BUNDLE.get("hour_do_mean", {}))
_PAIR_HOUR     = _s2d(_BUNDLE.get("pair_hour_mean", {}))
_PAIR_DOW      = _s2d(_BUNDLE.get("pair_dow_mean", {}))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def predict(request: dict) -> float:
    pu = int(request["pickup_zone"])
    do = int(request["dropoff_zone"])
    ts = datetime.fromisoformat(request["requested_at"])
    pax = int(request.get("passenger_count", 1))
    pax = max(1, min(9, pax)) if pax > 0 else 1

    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    week = ts.isocalendar()[1]

    pu_lat = float(_META.at[pu, "lat"]) if pu in _META.index else 40.75
    pu_lon = float(_META.at[pu, "lon"]) if pu in _META.index else -73.98
    do_lat = float(_META.at[do, "lat"]) if do in _META.index else 40.75
    do_lon = float(_META.at[do, "lon"]) if do in _META.index else -73.98

    pu_boro = int(_BOROUGH_CATS.get(pu, -1))
    do_boro = int(_BOROUGH_CATS.get(do, -1))
    dist = _haversine_km(pu_lat, pu_lon, do_lat, do_lon)

    pair = (pu, do)
    if pair in _PAIR_DICT:
        pm, pmed, pcount = _PAIR_DICT[pair]
        pair_mean = float(pm)
        pair_median = float(pmed)
        log_pair_count = float(np.log1p(pcount))
    else:
        pair_mean = float(_PU_MEAN.get(pu, _GLOBAL_MEAN))
        pair_median = float(_GLOBAL_MEDIAN)
        log_pair_count = 0.0

    pu_zone_mean = float(_PU_MEAN.get(pu, _GLOBAL_MEAN))
    do_zone_mean = float(_DO_MEAN.get(do, _GLOBAL_MEAN))

    row = {
        "hour": hour,
        "dow": dow,
        "month": month,
        "week_of_year": week,
        "day_of_month": ts.day,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_rush_hour": int((7 <= hour <= 9 or 16 <= hour <= 19) and dow < 5),
        "is_weekend": int(dow >= 5),
        "is_late_night": int(hour <= 4),
        "passenger_count": pax,
        "pickup_zone": pu,
        "dropoff_zone": do,
        "pickup_borough": pu_boro,
        "dropoff_borough": do_boro,
        "haversine_km": dist,
        "is_same_zone": int(pu == do),
        "same_borough": int(pu_boro == do_boro),
        "is_airport_pickup": int(pu in _AIRPORTS),
        "is_airport_dropoff": int(do in _AIRPORTS),
        "pair_mean": pair_mean,
        "pair_median": pair_median,
        "log_pair_count": log_pair_count,
        "pair_hour_mean": float(_PAIR_HOUR.get((pu, do, hour), pair_mean)),
        "pair_dow_mean":  float(_PAIR_DOW.get((pu, do, dow),   pair_mean)),
        "pickup_zone_mean": pu_zone_mean,
        "dropoff_zone_mean": do_zone_mean,
        "hour_pu_mean": float(_HOUR_PU_MEAN.get((hour, pu), pu_zone_mean)),
        "hour_do_mean": float(_HOUR_DO_MEAN.get((hour, do), do_zone_mean)),
    }

    features = pd.DataFrame([row])[_FEATURE_COLS]
    lgbm_pred = float(_MODEL.predict(features)[0])
    final = _BLEND_W * lgbm_pred + (1 - _BLEND_W) * pair_mean
    return float(max(final, 30.0))
