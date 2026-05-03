"""Submission interface — ensemble of XGBoost + LightGBM + NN + lookup.

Dev MAE: 255.9 s  (vs 258.5 s LightGBM CPU baseline)
Blend:   xgb=0.180  lgb=0.320  nn=0.260  lookup=0.240
"""

from __future__ import annotations
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
import lightgbm as lgb

_DIR = Path(__file__).parent

# Load slim bundle (no xgb/lgb objects — loaded from files below)
with open(_DIR / "model.pkl", "rb") as _f:
    _B = pickle.load(_f)

# Weights and config
_WX, _WC, _WN, _WL = _B["blend_weights"]
_FEAT  = _B["feature_names"]
_GM    = _B["global_mean"]
_GMD   = _B["global_median"]
_GP75  = _B.get("global_p75", _GM)
_META  = _B["zone_meta"].set_index("LocationID")
_BC    = _B["borough_cats"]          # dict zone→boro_code
_AIRPORTS = frozenset({1, 132, 138})

# Lookup dicts
_PAIR_AGG   = {(int(r.pickup_zone), int(r.dropoff_zone)):
               (float(r.pair_mean), float(r.pair_median), float(r.pair_std), float(r.pair_count))
               for r in _B["pair_agg"].reset_index().itertuples()}
_PU_MEAN    = _B["pu_mean"]
_DO_MEAN    = _B["do_mean"]
_HOUR_PU    = _B["hour_pu_mean"]
_HOUR_DO    = _B["hour_do_mean"]
_PAIR_HOUR  = _B["pair_hour_mean"]
_PAIR_DOW   = _B["pair_dow_mean"]
_PAIR_MONTH = _B["pair_month_mean"]
_BORO_PAIR  = _B["boro_pair_mean"]

# XGBoost — load from binary file (cross-platform safe)
_XGB = xgb.Booster()
_XGB.load_model(str(_DIR / "xgb_model.ubj"))

# LightGBM — load from text file (cross-platform safe)
# Must load before importing torch — macOS ARM OpenMP dylib conflict
_LGB = lgb.Booster(model_file=str(_DIR / "lgb_model.txt"))

# torch imported after lgb to avoid macOS ARM OpenMP conflict
import torch
import torch.nn as nn

# NN
_N_CONT  = 34
_N_ZONES = 266

class _ETANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pu_emb   = nn.Embedding(_N_ZONES, 16, padding_idx=0)
        self.do_emb   = nn.Embedding(_N_ZONES, 16, padding_idx=0)
        self.boro_emb = nn.Embedding(8, 4)
        self.hour_emb = nn.Embedding(24, 6)
        self.dow_emb  = nn.Embedding(7, 4)
        in_dim = 16+16+4+4+6+4 + _N_CONT + 1   # 85
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, cont, pu, do, pub, dob, h, dow, xgb_pred):
        x = torch.cat([
            self.pu_emb(pu), self.do_emb(do),
            self.boro_emb(pub), self.boro_emb(dob),
            self.hour_emb(h), self.dow_emb(dow),
            cont, xgb_pred.unsqueeze(1),
        ], dim=1)
        return self.net(x).squeeze(1)

_NN = _ETANet()
_NN.load_state_dict(_B["nn_state"])
_NN.eval()


def _haversine(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(float(np.clip(a, 0.0, 1.0)))))


def _get_latlon(zone):
    DEFAULT_LAT, DEFAULT_LON = 40.7128, -74.006
    if zone not in _META.index:
        return DEFAULT_LAT, DEFAULT_LON
    lat = float(_META.at[zone, "lat"]); lon = float(_META.at[zone, "lon"])
    if np.isnan(lat) or np.isnan(lon):
        return DEFAULT_LAT, DEFAULT_LON
    return lat, lon


def _build_row(request: dict) -> tuple:
    pu  = int(request["pickup_zone"])
    do  = int(request["dropoff_zone"])
    pax = max(1, min(9, int(request.get("passenger_count", 1))))
    ts  = datetime.fromisoformat(request["requested_at"])

    h  = ts.hour; dw = ts.weekday(); mo = ts.month
    dy = ts.day;  wk = ts.isocalendar()[1]

    pub = max(0, min(7, int(_BC.get(pu, -1)) + 1))
    dob = max(0, min(7, int(_BC.get(do, -1)) + 1))

    if (pu, do) in _PAIR_AGG:
        pm, pmed, pstd, pcnt = _PAIR_AGG[(pu, do)]
    else:
        pm, pmed, pstd, pcnt = float(_PU_MEAN.get(pu, _GM)), _GMD, _GP75, 0.0

    ph  = float(_PAIR_HOUR.get((pu, do, h),  pm))
    pdw = float(_PAIR_DOW.get((pu, do, dw),  pm))
    pmo = float(_PAIR_MONTH.get((pu, do, mo), pm))
    puz = float(_PU_MEAN.get(pu, _GM))
    doz = float(_DO_MEAN.get(do, _GM))
    hpu = float(_HOUR_PU.get((h, pu), puz))
    hdo = float(_HOUR_DO.get((h, do), doz))
    bpm = float(_BORO_PAIR.get((pub, dob), _GM))

    pu_lat, pu_lon = _get_latlon(pu)
    do_lat, do_lon = _get_latlon(do)
    dist      = _haversine(pu_lat, pu_lon, do_lat, do_lon)
    manhattan = (abs(do_lat - pu_lat) + abs(do_lon - pu_lon)) * 111.0

    is_christmas = float(mo == 12 and 23 <= dy <= 26)
    is_nye       = float(mo == 12 and dy >= 30)

    row = [
        pm, pmed, pstd, float(np.log1p(pcnt)),
        ph, pdw, pmo,
        puz, doz, hpu, hdo, bpm,
        dist, manhattan,
        float(h), float(dw), float(mo), float(dy), float(wk),
        float(np.sin(2*np.pi*h/24)), float(np.cos(2*np.pi*h/24)), float(dw/6.0),
        float(pu), float(do), float(pub), float(dob),
        float(pu == do), float(pub == dob),
        float(pu in _AIRPORTS), float(do in _AIRPORTS),
        max(is_christmas, is_nye), is_christmas, is_nye,
        float(pax),
    ]
    return row, pm, pu, do, pub, dob, h, dw


def predict(request: dict) -> float:
    row, pm, pu, do, pub, dob, h, dw = _build_row(request)
    X = np.array([row], dtype="float32")

    xgb_pred = float(_XGB.predict(xgb.DMatrix(X, feature_names=_FEAT))[0])
    lgb_pred = float(_LGB.predict(X)[0])

    with torch.no_grad():
        nn_pred = float(_NN(
            torch.from_numpy(X),
            torch.tensor([min(max(pu, 1), 265)], dtype=torch.long),
            torch.tensor([min(max(do, 1), 265)], dtype=torch.long),
            torch.tensor([pub], dtype=torch.long),
            torch.tensor([dob], dtype=torch.long),
            torch.tensor([h],   dtype=torch.long),
            torch.tensor([dw],  dtype=torch.long),
            torch.tensor([xgb_pred], dtype=torch.float32),
        ).item())

    final = _WX*xgb_pred + _WC*lgb_pred + _WN*nn_pred + _WL*pm
    return float(max(final, 30.0))
