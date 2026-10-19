"""Run the local Ames estimator portal with Python's built-in web server."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "archive" / "train.csv"
PORTAL = ROOT / "portal.html"
MODEL = None
COLUMNS = None
# FHFA/FRED Ames MSA all-transactions HPI: 2026 Q2 / 2008 Q2.
# The form omits YrSold, so the fitted pipeline imputes its training median (2008).
MARKET_ADJUSTMENT = 324.78 / 167.07
QUALITY = {"1": 2, "2": 4, "3": 5, "4": 7, "5": 9}
NEIGHBORHOODS = {
    "Blmngtn":"Bloomington Heights", "Blueste":"Bluestem", "BrDale":"Briardale",
    "BrkSide":"Brookside", "ClearCr":"Clear Creek", "CollgCr":"College Creek",
    "Crawfor":"Crawford", "Edwards":"Edwards", "Gilbert":"Gilbert",
    "IDOTRR":"Iowa DOT and Rail Road", "MeadowV":"Meadow Village", "Mitchel":"Mitchell",
    "Names":"North Ames", "NoRidge":"Northridge", "NPkVill":"Northpark Villa",
    "NridgHt":"Northridge Heights", "NWAmes":"Northwest Ames", "OldTown":"Old Town",
    "SWISU":"South & West of Iowa State University", "Sawyer":"Sawyer",
    "SawyerW":"Sawyer West", "Somerst":"Somerset", "StoneBr":"Stone Brook",
    "Timber":"Timberland", "Veenker":"Veenker",
}

def load_model():
    global MODEL, COLUMNS
    if MODEL is None:
        homes = pd.read_csv(DATA)
        y = homes["SalePrice"].astype(float)
        X = homes.drop(columns=["SalePrice", "Id"], errors="ignore")
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
        prep = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
        ], sparse_threshold=0.0)
        estimator = TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64,32), activation="relu", solver="adam", alpha=.001,
                learning_rate_init=.002, max_iter=300, early_stopping=True, n_iter_no_change=18,
                validation_fraction=.12, random_state=42),
            transformer=StandardScaler(),
        )
        MODEL = Pipeline([("prep", prep), ("model", estimator)]).fit(X, y)
        COLUMNS = X.columns.tolist()
    return MODEL, COLUMNS

def predict(payload):
    model, columns = load_model()
    kind = payload["kind"]
    quality = QUALITY[payload["quality"]]
    finish = {2:"Fa", 4:"TA", 5:"TA", 7:"Gd", 9:"Ex"}[quality]
    area = int(payload["area"])
    two_story = kind == "two-story"
    row = {c: np.nan for c in columns}
    row.update({
        "MSSubClass": {"detached":20,"two-story":60,"townhome":120,"duplex":90}[kind],
        "Neighborhood": payload["neighborhood"], "OverallQual":quality,
        "OverallCond":QUALITY[payload["condition"]], "GrLivArea":area,
        "BedroomAbvGr":int(payload["beds"]), "FullBath":int(payload["baths"]), "HalfBath":0,
        "YearBuilt":int(payload["year"]), "YearRemodAdd":int(payload["year"]),
        "GarageCars":2 if payload["garage"] else 0, "GarageArea":440 if payload["garage"] else 0,
        "HouseStyle":"2Story" if two_story else "1Story",
        "BldgType":{"detached":"1Fam","two-story":"1Fam","townhome":"TwnhsE","duplex":"Duplx"}[kind],
        "1stFlrSF":int(area*(.58 if two_story else 1)), "2ndFlrSF":int(area*.42) if two_story else 0,
        "TotalBsmtSF":int(area*.55), "KitchenQual":finish, "ExterQual":finish,
    })
    return max(0, float(model.predict(pd.DataFrame([row], columns=columns))[0]))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404); return
        body = PORTAL.read_bytes()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            historical = predict(json.loads(self.rfile.read(size)))
            result = {
                "historical_estimate": historical,
                "market_adjusted_estimate": historical * MARKET_ADJUSTMENT,
                "market_adjustment": MARKET_ADJUSTMENT,
            }
            body = json.dumps(result).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, fmt, *args):
        print("[portal] " + fmt % args, flush=True)

if __name__ == "__main__":
    print("Ames estimator available at http://127.0.0.1:8501", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8501), Handler).serve_forever()
