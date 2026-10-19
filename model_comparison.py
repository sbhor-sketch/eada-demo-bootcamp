#!/usr/bin/env python3
"""Train small house-price models and write an HTML validation report."""

from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "Data" / "archive" / "train.csv"
OUT_FILE = ROOT / "house_price_model_report.html"
RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, num_cols), ("categorical", categorical, cat_cols)],
        sparse_threshold=0.0,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.maximum(y_pred, 0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
    }


def build_report(results: list[dict], dataset: dict, holdout: dict) -> str:
    successful = [r for r in results if r["status"] == "ok"]
    best = min(successful, key=lambda r: r["metrics"]["rmse"])
    rows = []
    for r in sorted(successful, key=lambda item: item["metrics"]["rmse"]):
        m = r["metrics"]
        rows.append(
            "<tr><td><strong>{}</strong>{}</td><td>${:,.0f}</td><td>${:,.0f}</td>"
            "<td>{:.3f}</td><td>{:.4f}</td><td>{:.2f}s</td></tr>".format(
                html.escape(r["name"]),
                '<span class="badge">Best RMSE</span>' if r["name"] == best["name"] else "",
                m["mae"], m["rmse"], m["r2"], m["rmsle"], r["seconds"],
            )
        )
    failed = [r for r in results if r["status"] != "ok"]
    for r in failed:
        rows.append(f'<tr class="unavailable"><td>{html.escape(r["name"])}</td><td colspan="5">{html.escape(r["note"])}</td></tr>')
    bar_rows = []
    max_mae = max((r["metrics"]["mae"] for r in successful), default=1)
    for r in sorted(successful, key=lambda item: item["metrics"]["mae"]):
        width = max(3, round(r["metrics"]["mae"] / max_mae * 100))
        bar_rows.append(
            '<div class="bar-row"><span>{}</span><div class="bar-track"><div class="bar" style="width:{}%"></div></div><b>${:,.0f}</b></div>'.format(
                html.escape(r["name"]), width, r["metrics"]["mae"]
            )
        )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in dataset["notes"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>House Price Model Report</title><style>
:root{{--ink:#172033;--muted:#65718a;--line:#e4e9f0;--blue:#3763e8;--bg:#f5f7fb;--green:#087e62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1120px;margin:40px auto;padding:0 24px}}header{{background:linear-gradient(120deg,#14254a,#315de3);color:white;padding:34px 38px;border-radius:20px;box-shadow:0 16px 40px #20356622}}
h1{{margin:0 0 8px;font-size:32px;letter-spacing:-.5px}}header p{{margin:0;color:#dce5ff}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}
.card,.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px}}.card .label{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}.card strong{{display:block;font-size:22px;margin-top:6px}}
.panel{{margin:16px 0;padding:24px}}h2{{font-size:19px;margin:0 0 14px}}.sub{{color:var(--muted);font-size:13px;margin:-8px 0 18px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:13px 12px;border-bottom:1px solid var(--line);text-align:right}}th{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}th:first-child,td:first-child{{text-align:left}}tr:last-child td{{border-bottom:0}}.badge{{font-size:10px;color:var(--green);background:#e6f6f1;padding:3px 7px;border-radius:20px;margin-left:8px;white-space:nowrap}}
.bar-row{{display:grid;grid-template-columns:245px 1fr 105px;align-items:center;gap:12px;margin:11px 0;font-size:13px}}.bar-track{{height:11px;background:#eef1f7;border-radius:10px;overflow:hidden}}.bar{{height:100%;background:linear-gradient(90deg,#6e8aff,#315de3);border-radius:10px}}
.unavailable td{{color:#8a5260;background:#fff9fa}}.callout{{border-left:4px solid #e8a23a;background:#fff9ed;padding:14px 16px;border-radius:6px;color:#554628}}.foot{{color:var(--muted);font-size:12px;padding:4px 3px 28px}}
@media(max-width:760px){{.cards{{grid-template-columns:repeat(2,1fr)}}.panel{{overflow-x:auto}}.bar-row{{grid-template-columns:145px 1fr 78px}}header{{padding:26px}}h1{{font-size:26px}}}}
</style></head><body><div class="wrap"><header><h1>House Price Model Report</h1>
<p>Small-model benchmark on the Ames Housing dataset · validation results on unseen homes</p></header>
<div class="cards"><div class="card"><span class="label">Best model (RMSE)</span><strong>{html.escape(best['name'])}</strong></div>
<div class="card"><span class="label">Best validation RMSE</span><strong>${best['metrics']['rmse']:,.0f}</strong></div>
<div class="card"><span class="label">Best validation MAE</span><strong>${best['metrics']['mae']:,.0f}</strong></div>
<div class="card"><span class="label">Validation R²</span><strong>{best['metrics']['r2']:.3f}</strong></div></div>
<section class="panel"><h2>Model leaderboard</h2><p class="sub">Sorted by RMSE (lower is better). All models use the same fixed 20% holdout split.</p>
<table><thead><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th><th>RMSLE ↓</th><th>Fit time</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="panel"><h2>Mean absolute error</h2><p class="sub">Typical absolute miss on the validation homes, in dollars.</p>{''.join(bar_rows)}</section>
<section class="panel"><h2>How to read the KPIs</h2><ul><li><b>MAE</b>: average absolute dollar error; easy to interpret.</li><li><b>RMSE</b>: penalizes large misses more heavily; used to rank the models above.</li><li><b>R²</b>: fraction of price variation explained (closer to 1 is better).</li><li><b>RMSLE</b>: error on log prices; useful when relative price errors matter.</li></ul></section>
<section class="panel"><h2>Data and evaluation</h2><ul><li>{dataset['rows']:,} labeled homes, {dataset['features']} input columns, target: SalePrice.</li><li>Random 80/20 train/validation split; random seed {RANDOM_STATE}. Validation rows were not used to fit models.</li><li>Missing values are imputed; categorical features are one-hot encoded. Input ID is excluded.</li><li>Neural network is a small 64→32 MLP with early stopping. Tree ensembles use limited depth/estimators for fast training.</li>{notes}</ul></section>
<p class="foot">Generated by model_comparison.py. This is a single holdout benchmark, not a guarantee of future market performance. Compare on the same split when changing the workflow.</p>
</div></body></html>'''


def main() -> None:
    if len(sys.argv) > 1:
        global TRAIN_FILE, OUT_FILE
        TRAIN_FILE = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) > 2:
        OUT_FILE = Path(sys.argv[2]).expanduser().resolve()
    df = pd.read_csv(TRAIN_FILE)
    if "SalePrice" not in df.columns:
        raise ValueError("Training CSV must contain a SalePrice target column")
    y = df["SalePrice"].astype(float).to_numpy()
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Neural Network (MLP)", TransformedTargetRegressor(regressor=MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.001, learning_rate_init=0.002, max_iter=300, early_stopping=True, n_iter_no_change=18, validation_fraction=0.12, random_state=RANDOM_STATE), transformer=StandardScaler())),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Hist Gradient Boosting", HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, learning_rate=0.08, l2_regularization=1.0, early_stopping=True, random_state=RANDOM_STATE)),
    ]
    xgb_note = None
    try:
        from xgboost import XGBRegressor

        models.append(("XGBoost", XGBRegressor(n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, objective="reg:squarederror", n_jobs=2, random_state=RANDOM_STATE, verbosity=0)))
    except Exception as exc:
        xgb_note = f"Not run: XGBoost could not load its native runtime ({type(exc).__name__}). Hist Gradient Boosting is included as a fast boosting alternative."
    results = []
    for name, estimator in models:
        start = time.perf_counter()
        model = Pipeline([("prep", make_preprocessor(X_train)), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        elapsed = time.perf_counter() - start
        results.append({"name": name, "status": "ok", "metrics": metrics(y_val, pred), "seconds": elapsed})
        print(f"{name}: RMSE ${results[-1]['metrics']['rmse']:,.0f} ({elapsed:.1f}s)", flush=True)

    if xgb_note:
        results.append({"name": "XGBoost", "status": "unavailable", "note": xgb_note})
    dataset = {
        "rows": len(df),
        "features": X.shape[1],
        "notes": [xgb_note] if xgb_note else ["XGBoost trained successfully in this environment."],
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(build_report(results, dataset, {"rows": len(y_val)}), encoding="utf-8")
    json_file = OUT_FILE.with_suffix(".json")
    json_file.write_text(json.dumps({"seed": RANDOM_STATE, "validation_rows": len(y_val), "results": results}, indent=2), encoding="utf-8")
    print(f"HTML report: {OUT_FILE}")
    print(f"Metrics data: {json_file}")


if __name__ == "__main__":
    main()
