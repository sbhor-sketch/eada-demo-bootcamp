"""Export the trained Ames pipeline as a self-contained browser predictor."""
from __future__ import annotations

import json
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
OUT = ROOT / "prediction_model.js"

homes = pd.read_csv(DATA)
y = homes["SalePrice"].astype(float)
X = homes.drop(columns=["SalePrice", "Id"], errors="ignore")
numeric = X.select_dtypes(include=[np.number]).columns.tolist()
categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
prep = ColumnTransformer([
    ("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ]), categorical),
], sparse_threshold=0.0)
estimator = TransformedTargetRegressor(
    regressor=MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.001,
        learning_rate_init=0.002, max_iter=300, early_stopping=True,
        n_iter_no_change=18, validation_fraction=0.12, random_state=42,
    ),
    transformer=StandardScaler(),
)
pipeline = Pipeline([("prep", prep), ("model", estimator)]).fit(X, y)

numeric_pipe = pipeline.named_steps["prep"].named_transformers_["numeric"]
imputer = numeric_pipe.named_steps["impute"]
scaler = numeric_pipe.named_steps["scale"]
categorical_pipe = pipeline.named_steps["prep"].named_transformers_["categorical"]
cat_imputer = categorical_pipe.named_steps["impute"]
encoder = categorical_pipe.named_steps["encode"]

cat_specs = []
for name, categories, mode, infrequent in zip(
    categorical,
    encoder.categories_,
    cat_imputer.statistics_,
    encoder.infrequent_categories_,
):
    infrequent_values = set(infrequent.tolist()) if infrequent is not None else set()
    outputs = [str(value) for value in categories if value not in infrequent_values]
    if infrequent_values:
        outputs.append("infrequent_sklearn")
    cat_specs.append({
        "name": name,
        "categories": [str(value) for value in categories],
        "infrequent": [str(value) for value in infrequent_values],
        "mode": str(mode),
        "outputs": outputs,
    })

target_scaler = pipeline.named_steps["model"].transformer_
mlp = pipeline.named_steps["model"].regressor_
spec = {
    "numeric": numeric,
    "medians": imputer.statistics_.tolist(),
    "indicatorFeatures": imputer.indicator_.features_.tolist(),
    "numericMeans": scaler.mean_.tolist(),
    "numericScales": scaler.scale_.tolist(),
    "categorical": cat_specs,
    "weights": [array.tolist() for array in mlp.coefs_],
    "biases": [array.tolist() for array in mlp.intercepts_],
    "targetMean": float(target_scaler.mean_[0]),
    "targetScale": float(target_scaler.scale_[0]),
}

javascript = """/* Generated from the Ames housing training pipeline. */
const PRICE_MODEL = %s;
window.estimateHome = function(property) {
  const values = {
    MSSubClass: ({detached:20, 'two-story':60, townhome:120, duplex:90})[property.kind],
    Neighborhood: property.neighborhood,
    OverallQual: ({1:2,2:4,3:5,4:7,5:9})[property.quality],
    OverallCond: ({1:2,2:4,3:5,4:7,5:9})[property.condition],
    GrLivArea: property.area, BedroomAbvGr: property.beds, FullBath: property.baths,
    HalfBath: 0, YearBuilt: property.year, YearRemodAdd: property.year,
    GarageCars: property.garage ? 2 : 0, GarageArea: property.garage ? 440 : 0,
    HouseStyle: property.kind === 'two-story' ? '2Story' : '1Story',
    BldgType: ({detached:'1Fam','two-story':'1Fam',townhome:'TwnhsE',duplex:'Duplx'})[property.kind],
    '1stFlrSF': Math.round(property.area * (property.kind === 'two-story' ? .58 : 1)),
    '2ndFlrSF': property.kind === 'two-story' ? Math.round(property.area * .42) : 0,
    TotalBsmtSF: Math.round(property.area * .55),
    KitchenQual: ({2:'Fa',4:'TA',5:'TA',7:'Gd',9:'Ex'})[({1:2,2:4,3:5,4:7,5:9})[property.quality]],
    ExterQual: ({2:'Fa',4:'TA',5:'TA',7:'Gd',9:'Ex'})[({1:2,2:4,3:5,4:7,5:9})[property.quality]],
  };
  const numericRaw = PRICE_MODEL.numeric.map(name => values[name] ?? null);
  const num = numericRaw.map((v,i) => ((v === null ? PRICE_MODEL.medians[i] : Number(v)) - PRICE_MODEL.numericMeans[i]) / PRICE_MODEL.numericScales[i]);
  for (let j = 0; j < PRICE_MODEL.indicatorFeatures.length; j++) {
    const i = PRICE_MODEL.indicatorFeatures[j], ix = PRICE_MODEL.numeric.length + j;
    const indicator = numericRaw[i] === null ? 1 : 0;
    num.push((indicator - PRICE_MODEL.numericMeans[ix]) / PRICE_MODEL.numericScales[ix]);
  }
  const encoded = [];
  for (const feature of PRICE_MODEL.categorical) {
    let value = values[feature.name] ?? feature.mode;
    let output = String(value);
    if (!feature.categories.includes(output)) output = null;
    else if (feature.infrequent.includes(output)) output = 'infrequent_sklearn';
    for (const category of feature.outputs) encoded.push(output === category ? 1 : 0);
  }
  let activation = num.concat(encoded);
  for (let layer = 0; layer < PRICE_MODEL.weights.length; layer++) {
    const matrix = PRICE_MODEL.weights[layer], bias = PRICE_MODEL.biases[layer];
    const next = bias.map((b,j) => {
      let sum = b;
      for (let i = 0; i < activation.length; i++) sum += activation[i] * matrix[i][j];
      return layer < PRICE_MODEL.weights.length - 1 ? Math.max(0, sum) : sum;
    });
    activation = next;
  }
  return Math.max(0, activation[0] * PRICE_MODEL.targetScale + PRICE_MODEL.targetMean);
};
""" % json.dumps(spec, separators=(",", ":"))
OUT.write_text(javascript, encoding="utf-8")
print(f"Wrote {OUT.name} ({OUT.stat().st_size:,} bytes)")
