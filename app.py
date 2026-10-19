"""Friendly Ames home-price estimator built on the project housing model."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "Data" / "archive" / "train.csv"
DATABASE_FILE = ROOT / "ronald.sqlite3"
LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode((ROOT / "ronald-logo.svg").read_bytes()).decode("ascii")
HPI_Q2_BY_YEAR = {
    2008: 167.07, 2009: 168.52, 2010: 166.78, 2011: 162.23, 2012: 165.05,
    2013: 169.35, 2014: 176.23, 2015: 183.12, 2016: 191.59, 2017: 199.87,
    2018: 207.07, 2019: 212.38, 2020: 217.26, 2021: 236.32, 2022: 272.07,
    2023: 287.38, 2024: 306.93, 2025: 307.62, 2026: 324.78,
}


def connect_db() -> sqlite3.Connection:
    db = sqlite3.connect(DATABASE_FILE)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


@contextmanager
def database():
    db = connect_db()
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize_database() -> None:
    with database() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                home_type TEXT NOT NULL,
                neighborhood TEXT NOT NULL,
                price INTEGER NOT NULL,
                area INTEGER NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                year_built INTEGER NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                photo BLOB,
                photo_mime TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
                sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                recipient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS listing_photos (
                id INTEGER PRIMARY KEY,
                listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
                image_data BLOB NOT NULL,
                position INTEGER NOT NULL DEFAULT 0
            );
        """)
        listing_columns = {row["name"] for row in db.execute("PRAGMA table_info(listings)")}
        if "photo" not in listing_columns:
            db.execute("ALTER TABLE listings ADD COLUMN photo BLOB")
        if "photo_mime" not in listing_columns:
            db.execute("ALTER TABLE listings ADD COLUMN photo_mime TEXT")


def normalize_listing_photo(uploaded_file) -> bytes:
    """Validate, resize, and strip metadata from an uploaded listing photo."""
    raw = uploaded_file.getvalue()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Each photo must be 5 MB or smaller.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > 20_000_000:
                raise ValueError("Photo dimensions are too large. Please choose a smaller image.")
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((1600, 1200))
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=84, optimize=True)
            return output.getvalue()
    except UnidentifiedImageError as error:
        raise ValueError("That file doesn’t look like a readable image.") from error
    except OSError as error:
        raise ValueError("We couldn’t read that image. Try a JPG, PNG, or WebP file.") from error


@st.cache_data(show_spinner=False)
def load_training_sales() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE)


def find_similar_sales(neighborhood: str, home_type: str, area: int, bedrooms: int, year_built: int) -> pd.DataFrame:
    """Find historical context sales near the entered home profile."""
    homes = load_training_sales()
    candidates = homes.loc[homes["Neighborhood"] == neighborhood].copy()
    if candidates.empty:
        candidates = homes.copy()
    dataset_type = {"Detached house": "1Fam", "Two-storey house": "1Fam", "Townhome": "TwnhsE", "Duplex": "Duplx"}[home_type]
    same_type = candidates.loc[candidates["BldgType"] == dataset_type]
    if not same_type.empty:
        candidates = same_type.copy()
    candidates["match_score"] = (
        (candidates["GrLivArea"] - area).abs() / 500
        + (candidates["BedroomAbvGr"] - bedrooms).abs() * 1.5
        + (candidates["YearBuilt"] - year_built).abs() / 30
    )
    return candidates.sort_values("match_score").head(3)


def password_digest(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000).hex()


def create_account(email: str, password: str) -> int:
    email = email.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("Enter an email address in the usual format. We won’t verify it by email yet.")
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")
    salt = secrets.token_bytes(16)
    with database() as db:
        cursor = db.execute(
            "INSERT INTO users(email, salt, password_hash) VALUES (?, ?, ?)",
            (email, salt.hex(), password_digest(password, salt)),
        )
        return int(cursor.lastrowid)


def authenticate(email: str, password: str) -> int | None:
    with database() as db:
        user = db.execute("SELECT id, salt, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if user is None:
        return None
    actual = password_digest(password, bytes.fromhex(user["salt"]))
    return int(user["id"]) if hmac.compare_digest(actual, user["password_hash"]) else None


def current_user() -> sqlite3.Row | None:
    user_id = st.session_state.get("ronald_user_id")
    if not user_id:
        return None
    with database() as db:
        return db.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()


initialize_database()
try:
    DATABASE_FILE.chmod(0o600)
except OSError:
    pass

NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "Names": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}

# Approximate neighborhood center points, geocoded for Ames, Iowa. They are
# useful for orientation, but are not official neighborhood boundaries.
NEIGHBORHOOD_COORDS = {
    "Blmngtn": (42.056376, -93.644471), "Blueste": (42.023090, -93.663783),
    "BrDale": (42.052795, -93.628821), "BrkSide": (42.028653, -93.630386),
    "ClearCr": (42.036097, -93.648830), "CollgCr": (42.022197, -93.651510),
    "Crawfor": (42.018614, -93.648898), "Edwards": (42.015510, -93.685391),
    "Gilbert": (42.0430, -93.6500), "IDOTRR": (42.021997, -93.622018),
    "MeadowV": (41.992448, -93.602645), "Mitchel": (41.990308, -93.601053),
    "Names": (42.049358, -93.622442), "NoRidge": (42.047831, -93.646745),
    "NPkVill": (42.030781, -93.631913), "NridgHt": (42.059773, -93.649471),
    "NWAmes": (42.038277, -93.625770), "OldTown": (42.029046, -93.614340),
    "SWISU": (42.026657, -93.646452), "Sawyer": (42.033903, -93.677066),
    "SawyerW": (42.024906, -93.659599), "Somerst": (42.052628, -93.644582),
    "StoneBr": (42.059727, -93.637642), "Timber": (41.999973, -93.649693),
    "Veenker": (42.041304, -93.650302),
}

QUALITY = {
    "1 · Needs a lot of work": 2,
    "2 · Below average": 4,
    "3 · Comfortable": 5,
    "4 · Very good": 7,
    "5 · Premium finish": 9,
}


@st.cache_resource(show_spinner="Getting the home estimator ready…")
def load_model() -> tuple[Pipeline, list[str]]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")
    homes = pd.read_csv(TRAIN_FILE)
    y = homes["SalePrice"].astype(float)
    X = homes.drop(columns=["SalePrice", "Id"], errors="ignore")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    estimator = TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.002,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=18,
            validation_fraction=0.12,
            random_state=42,
        ),
        transformer=StandardScaler(),
    )
    model = Pipeline([("prep", prep), ("model", estimator)])
    model.fit(X, y)
    return model, X.columns.tolist()


def make_property(
    columns: list[str],
    *,
    home_type: str,
    neighborhood: str,
    quality: str,
    condition: str,
    area: int,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    garage: bool,
) -> pd.DataFrame:
    """Translate plain-language answers into the dataset's original columns."""
    row = {column: np.nan for column in columns}
    two_story = home_type == "Two-storey house"
    row.update(
        {
            "MSSubClass": {"Detached house": 20, "Two-storey house": 20, "Townhome": 120, "Duplex": 90}[home_type],
            "Neighborhood": neighborhood,
            "OverallQual": QUALITY[quality],
            "OverallCond": QUALITY[condition],
            "GrLivArea": area,
            "BedroomAbvGr": bedrooms,
            "FullBath": bathrooms,
            "HalfBath": 0,
            "YearBuilt": year_built,
            "YearRemodAdd": year_built,
            "GarageCars": 2 if garage else 0,
            "GarageArea": 440 if garage else 0,
            "HouseStyle": "2Story" if two_story else "1Story",
            "BldgType": {"Detached house": "1Fam", "Two-storey house": "1Fam", "Townhome": "TwnhsE", "Duplex": "Duplx"}[home_type],
            "1stFlrSF": int(area * (0.58 if two_story else 1)),
            "2ndFlrSF": int(area * 0.42) if two_story else 0,
            "TotalBsmtSF": int(area * 0.55),
            "KitchenQual": {2: "Fa", 4: "TA", 5: "TA", 7: "Gd", 9: "Ex"}[QUALITY[quality]],
            "ExterQual": {2: "Fa", 4: "TA", 5: "TA", 7: "Gd", 9: "Ex"}[QUALITY[quality]],
        }
    )
    return pd.DataFrame([row], columns=columns)


st.set_page_config(page_title="Ronald · Ames homes", page_icon="🏠", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#202923; --muted:#727a70; --green:#356b4c; --paper:#f6f7f2; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    .block-container { max-width:1120px; padding-top:2.25rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
    .eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#57735c; }
    .hero { padding:2rem 2.1rem 1.7rem; border-radius:24px; background:linear-gradient(125deg,#e4ebdc,#f2ead9); margin-bottom:1.3rem; }
    .hero h1 { font-size:clamp(2rem,4vw,3.2rem); margin:.35rem 0 .45rem; color:#25372a; }
    .hero p { color:#526153; font-size:1.05rem; max-width:650px; margin:0; }
    .section-card { border:1px solid #e4e7dd; background:#fff; border-radius:18px; padding:1.25rem 1.4rem; margin:0 0 1rem; }
    .estimate { background:#2d543a; color:white; padding:1.5rem; border-radius:20px; }
    .estimate .amount { font-family:'Manrope',sans-serif; font-size:2.5rem; font-weight:800; letter-spacing:-.04em; line-height:1.1; }
    .estimate small { color:#d5e3d5; }
    div[data-testid="stButton"] button[kind="primary"] { background:#356b4c; border-color:#356b4c; border-radius:12px; min-height:3rem; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background:#294f38; border-color:#294f38; }
    div[data-testid="stButton"] button[kind="secondary"] { background:#fff; color:#294f38; border:1px solid #d8e0d4; border-radius:12px; min-height:3rem; }
    div[data-testid="stButton"] button[kind="secondary"]:hover { background:#e9efe5; border-color:#9db39f; color:#294f38; }
    .note { color:#727a70; font-size:.85rem; }
    .site-brand { display:flex; align-items:center; gap:.7rem; color:#294f38; font:800 1.2rem Manrope,sans-serif; }
    .site-logo { display:block; width:2.65rem; height:2.65rem; filter:drop-shadow(0 3px 4px #1d452422); }
    .site-brand small { display:block; color:#727a70; font:500 .75rem 'DM Sans',sans-serif; letter-spacing:0; }
    .welcome-hero { display:grid; grid-template-columns:1.2fr .8fr; align-items:center; gap:2rem; padding:2.5rem 2.7rem; border-radius:28px; background:linear-gradient(125deg,#e4ebdc,#f2ead9); margin:1rem 0 1.3rem; overflow:hidden; }
    .welcome-hero h1 { font-size:clamp(2.4rem,5vw,4.25rem); line-height:1.03; margin:.5rem 0 1rem; color:#25372a; }
    .welcome-hero p { color:#526153; font-size:1.1rem; max-width:560px; }
    .welcome-art { min-height:260px; border-radius:24px; background:linear-gradient(150deg,#d4e2cf,#f7e4c6); display:flex; align-items:center; justify-content:center; }
    .category-card { background:white; border:1px solid #e4e7dd; border-radius:20px; overflow:hidden; height:100%; }
    .category-art { height:145px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#e9efe5,#f3eadc); }
    .category-art svg { width:100%; height:145px; }
    .category-copy { padding:1rem 1.1rem 1.2rem; }
    .category-copy h3 { margin:.1rem 0 .35rem; color:#294f38; }
    .category-copy p { color:#727a70; font-size:.9rem; min-height:2.8rem; }
    .welcome-note { color:#727a70; font-size:.82rem; }
    @media(max-width:760px){ .welcome-hero{grid-template-columns:1fr;padding:1.5rem}.welcome-art{min-height:180px} }
    </style>
    """,
    unsafe_allow_html=True,
)

if "ronald_page" not in st.session_state:
    st.session_state.ronald_page = "Welcome"
if "ronald_home_filter" not in st.session_state:
    st.session_state.ronald_home_filter = "All home types"
page_names = ["Welcome", "Estimate", "Explore homes", "List a home", "Account"]
nav_cols = st.columns([1.5, 1, 1, 1.35, 1.15, 1.1])
with nav_cols[0]:
    st.markdown(f'<div class="site-brand"><img class="site-logo" src="{LOGO_DATA_URI}" alt="Ronald home logo"><span>Ronald<small>AMES, IOWA</small></span></div>', unsafe_allow_html=True)
for col, page_name, key in zip(nav_cols[1:], page_names, ["welcome", "estimate", "explore", "list", "account"]):
    with col:
        if st.button(page_name, key=f"nav_{key}", type="primary" if st.session_state.ronald_page == page_name else "secondary", use_container_width=True):
            st.session_state.ronald_page = page_name
            st.rerun()

current_page = st.session_state.ronald_page

if current_page == "Welcome":
    st.markdown('''
    <section class="welcome-hero">
      <div><div class="eyebrow">A simpler way to find your place</div><h1>Home starts with a feeling.</h1>
      <p>Explore homes around Ames, get a friendly idea of what a home could be worth, or share a place of your own.</p></div>
      <div class="welcome-art" aria-label="Illustration of a welcoming home">
        <svg viewBox="0 0 420 270" width="94%" height="240" role="img" aria-label="Illustrated home with garden">
          <circle cx="320" cy="58" r="30" fill="#f8dca2"/><path d="M35 228 Q120 191 205 225 T390 215 V270 H35Z" fill="#bdd0b5"/>
          <path d="M95 130 205 43 315 130v100H95Z" fill="#fffaf0"/><path d="m75 135 130-106 130 106-16 19-114-92-114 92Z" fill="#57775b"/>
          <path d="M177 230v-66q0-30 28-30t28 30v66Z" fill="#c98d5c"/><rect x="118" y="148" width="38" height="38" rx="4" fill="#c7dcd5"/><rect x="256" y="148" width="38" height="38" rx="4" fill="#c7dcd5"/>
          <path d="M137 148v38m-19-19h38m119-19v38m-19-19h38" stroke="#fffaf0" stroke-width="5"/>
          <path d="M65 227q-2-35 22-44 18 24 1 45m14 1q-3-28 18-37 16 22 2 39" fill="#75936d"/>
          <path d="M330 229q-4-33 17-43 20 20 5 44m12 0q-1-25 19-34 14 17 3 36" fill="#75936d"/>
        </svg>
      </div>
    </section>
    ''', unsafe_allow_html=True)
    st.markdown("### What are you looking for?")
    st.caption("Choose a home style to explore, or start with a price estimate.")
    category_cols = st.columns(3, gap="medium")
    categories = [
        ('''<svg viewBox="0 0 320 150" aria-hidden="true"><circle cx="262" cy="34" r="17" fill="#f4d38e"/><path d="M0 119 Q70 93 144 119 T320 112 V150 H0Z" fill="#bdd0b5"/><path d="M75 71 151 18 227 71v63H75Z" fill="#fffaf0"/><path d="m60 74 91-66 92 66-11 12-81-57-81 57Z" fill="#57775b"/><rect x="96" y="84" width="27" height="25" rx="3" fill="#c7dcd5"/><rect x="178" y="84" width="27" height="25" rx="3" fill="#c7dcd5"/><path d="M96 96h27m-14-12v25m69-13h27m-14-12v25" stroke="#fffaf0" stroke-width="3"/><path d="M138 134v-32q0-18 13-18t13 18v32Z" fill="#c98d5c"/><path d="M35 131q-2-27 15-34 16 17 5 35m227-1q-1-25 17-31 15 16 4 33" fill="#75936d"/></svg>''', "Detached homes", "Room to spread out, with a place to make your own.", "Detached house"),
        ('''<svg viewBox="0 0 320 150" aria-hidden="true"><circle cx="258" cy="32" r="17" fill="#f4d38e"/><path d="M0 121 Q80 97 160 119 T320 112 V150 H0Z" fill="#bdd0b5"/><path d="M42 62 93 29l50 33v74H42Zm93 0 50-33 50 33v74h-50V62Zm93 0 50-33 50 33v74h-50V62Z" fill="#fffaf0"/><path d="m31 65 62-44 61 44-10 10-51-36-52 36Zm93 0 62-44 61 44-10 10-51-36-52 36Zm93 0 62-44 61 44-10 10-51-36-52 36Z" fill="#57775b"/><path d="M57 83h20v22H57zm53 0h20v22h-20zm41 0h20v22h-20zm53 0h20v22h-20zm42 0h20v22h-20zm53 0h20v22h-20z" fill="#c7dcd5"/><path d="M82 136v-26q0-15 13-15t13 15v26m93 0v-26q0-15 13-15t13 15v26m93 0v-26q0-15 13-15t13 15v26" fill="#c98d5c"/></svg>''', "Townhomes", "Comfortable, connected living around the neighborhood.", "Townhome"),
        ('''<svg viewBox="0 0 320 150" aria-hidden="true"><circle cx="260" cy="33" r="17" fill="#f4d38e"/><path d="M0 120 Q74 98 153 118 T320 113 V150 H0Z" fill="#bdd0b5"/><path d="M48 70 117 24l68 46v65H48Z" fill="#fffaf0"/><path d="M148 70 216 24l68 46v65H148Z" fill="#f8f0df"/><path d="m35 73 82-59 82 59-11 12-71-51-71 51Zm100 0 81-59 82 59-11 12-71-51-70 51Z" fill="#57775b"/><rect x="65" y="84" width="28" height="25" rx="3" fill="#c7dcd5"/><rect x="238" y="84" width="28" height="25" rx="3" fill="#c7dcd5"/><path d="M64 96h29m-14-12v25m159-13h28m-14-12v25" stroke="#fffaf0" stroke-width="3"/><path d="M105 135v-28q0-16 13-16t13 16v28m73 0v-28q0-16 13-16t13 16v28" fill="#c98d5c"/><path d="M20 133q-1-22 14-28 14 14 4 29" fill="#75936d"/></svg>''', "Duplexes", "Flexible spaces for more than one household.", "Duplex"),
    ]
    for col, (illustration, title, description, home_filter) in zip(category_cols, categories):
        with col:
            st.markdown(f'<div class="category-card"><div class="category-art">{illustration}</div><div class="category-copy"><h3>{title}</h3><p>{description}</p></div></div>', unsafe_allow_html=True)
            if st.button(f"Explore {title.lower()}", key=f"category_{title}", use_container_width=True):
                st.session_state.ronald_home_filter = home_filter
                st.session_state.ronald_page = "Explore homes"
                st.rerun()
    st.write("")
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("Check a home’s value", type="primary", use_container_width=True):
            st.session_state.ronald_page = "Estimate"
            st.rerun()
    with action_cols[1]:
        if st.button("List your home", use_container_width=True):
            st.session_state.ronald_page = "List a home"
            st.rerun()
    with action_cols[2]:
        st.markdown('<p class="welcome-note">Ronald is a local home marketplace for Ames, Iowa. Listings are shared by community members.</p>', unsafe_allow_html=True)

elif current_page == "Estimate":
    st.markdown('<div class="hero"><div class="eyebrow">RONALD · HOME VALUE</div><h1>Get a starting estimate.</h1><p>Tell us a few easy details and see how Ames home prices have moved over time.</p></div>', unsafe_allow_html=True)

if current_page == "Estimate":
    left, right = st.columns([1.45, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Tell us about the home")
        st.caption("A few basics are enough. You can leave the finer details to the model.")
        c1, c2 = st.columns(2)
        with c1:
            home_type = st.selectbox("What kind of home is it?", ["Detached house", "Two-storey house", "Townhome", "Duplex"])
            neighborhood_name = st.selectbox("Which Ames neighborhood?", list(NEIGHBORHOODS.values()), index=list(NEIGHBORHOODS.values()).index("College Creek"))
            area = st.number_input("Finished living area", min_value=400, max_value=6000, value=1600, step=50, help="Approximate indoor living space, in square feet.")
            bedrooms = st.select_slider("Bedrooms", options=[1, 2, 3, 4, 5, 6], value=3)
        with c2:
            bathrooms = st.select_slider("Full bathrooms", options=[1, 2, 3, 4], value=2)
            year_built = st.number_input("Year built", min_value=1872, max_value=2010, value=2000, step=1, help="The source sales data runs through 2010, so the estimator is limited to that range.")
            garage = st.toggle("Has a garage", value=True)
            quality = st.select_slider("Overall finish", options=list(QUALITY), value="3 · Comfortable", help="Think about materials, finishes, and how polished the home feels.")
            condition = st.select_slider("Overall upkeep", options=list(QUALITY), value="3 · Comfortable", help="Think about repairs and maintenance, regardless of style or finishes.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Where is this neighborhood?")
        st.caption("Approximate neighborhood locations in Ames. Dots are markers, not official borders.")
        selected_label = next(label for label, full_name in NEIGHBORHOODS.items() if full_name == neighborhood_name)
        coords = pd.DataFrame([
            {"latitude": lat, "longitude": lon, "color": "#356b4c" if code == selected_label else "#97aa98"}
            for code, (lat, lon) in NEIGHBORHOOD_COORDS.items()
        ])
        st.map(coords, latitude="latitude", longitude="longitude", color="color", size=12, zoom=11)
        selected_lat, selected_lon = NEIGHBORHOOD_COORDS[selected_label]
        st.markdown(f"**Selected:** {neighborhood_name}  ·  [Open approximate location in Google Maps](https://www.google.com/maps/search/?api=1&query={selected_lat},{selected_lon})")
        with st.expander("Browse all 25 neighborhoods"):
            st.markdown(" · ".join(NEIGHBORHOODS.values()))
        st.markdown('<p class="note">Locations are approximate orientation points based on a geocoded neighborhood reference. The source data has neighborhood labels but does not include official boundary lines; location labels may not match current city association boundaries exactly. <a href="https://limbachia.github.io/house-price-prediction/00-data-wrangling.html" target="_blank">Coordinate reference</a>.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Your estimate")
        st.write("When you’re ready, we’ll compare these details with the model’s examples.")
        estimate_button = st.button("Estimate home price", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if estimate_button:
            try:
                model, columns = load_model()
                neighborhood_code = selected_label
                property_data = make_property(
                    columns,
                    home_type=home_type,
                    neighborhood=neighborhood_code,
                    quality=quality,
                    condition=condition,
                    area=int(area),
                    bedrooms=int(bedrooms),
                    bathrooms=int(bathrooms),
                    year_built=int(year_built),
                    garage=garage,
                )
                prediction = max(0, float(model.predict(property_data)[0]))
                baseline_index = HPI_Q2_BY_YEAR[2008]
                market_2026 = prediction * HPI_Q2_BY_YEAR[2026] / baseline_index
                total_change_pct = (HPI_Q2_BY_YEAR[2026] / baseline_index - 1) * 100
                st.markdown(
                    f'<div class="estimate"><small>Rough 2026 market-adjusted reference</small><div class="amount">${market_2026:,.0f}</div>'
                    f'<small>{neighborhood_name} · {area:,} sq ft · {bedrooms} bed · {bathrooms} bath</small></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Original model estimate: ${prediction:,.0f} in the historical dataset era · broad Ames market trend: {total_change_pct:+.1f}% since 2008")
                trend = pd.DataFrame({
                    "Year": list(HPI_Q2_BY_YEAR),
                    "Estimated price": [prediction * index / baseline_index for index in HPI_Q2_BY_YEAR.values()],
                })
                trend["Change from prior year"] = trend["Estimated price"].pct_change().mul(100)
                st.markdown("#### Estimated value trend")
                st.caption("Each point scales this home’s original estimate by the Ames FHFA house-price index (second quarter each year). Positive values mean the index rose; negative values mean it fell.")
                st.line_chart(trend, x="Year", y="Estimated price")
                trend_table = trend[["Year", "Estimated price", "Change from prior year"]].copy()
                trend_table["Change from prior year"] = trend_table["Change from prior year"].map(lambda value: "—" if pd.isna(value) else f"{value:+.1f}%")
                st.dataframe(
                    trend_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"Estimated price": st.column_config.NumberColumn(format="$%d")},
                )
                st.caption("The index is metro-wide, not property-specific. 2026 data is through Q2; this is a historical trend illustration, not a forecast of this home’s future sale price. Source: FHFA via FRED (ATNHPIUS11180Q).")
                st.caption("In the model’s 20% validation sample, its typical absolute error was about $18,000 (MAE). Actual sale prices can differ.")
            except Exception as error:
                st.error(f"We couldn’t create an estimate: {error}")
        else:
            st.info("Your estimate will appear here.")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("How to read this")
        st.markdown("This is an educational estimate from Ames home sales recorded from 2006 to 2010. It is not a formal appraisal or a current market quote, and the old sales data may not reflect today’s prices. The source data covers houses, townhomes, and duplexes—not apartments.")
        st.markdown('</div>', unsafe_allow_html=True)

if current_page == "Explore homes":
    st.markdown('<div class="hero"><div class="eyebrow">THE RONALD HOME COLLECTION</div><h1>Explore homes in Ames.</h1><p>Browse photos and details shared by local sellers.</p></div>', unsafe_allow_html=True)
    st.subheader("Homes listed on Ronald")
    st.caption("Browse homes shared by the community. Sign in to send a seller a private message.")
    st.selectbox("Filter by home type", ["All home types", "Detached house", "Two-storey house", "Townhome", "Duplex"], key="ronald_home_filter")
    with database() as db:
        if st.session_state.ronald_home_filter == "All home types":
            listings = db.execute("SELECT * FROM listings ORDER BY created_at DESC, id DESC").fetchall()
        else:
            listings = db.execute("SELECT * FROM listings WHERE home_type=? ORDER BY created_at DESC, id DESC", (st.session_state.ronald_home_filter,)).fetchall()
        photos_by_listing: dict[int, list[bytes]] = {}
        for photo in db.execute("SELECT listing_id, image_data FROM listing_photos ORDER BY position, id").fetchall():
            photos_by_listing.setdefault(photo["listing_id"], []).append(photo["image_data"])
    if not listings:
        st.info("No homes have been listed yet. Be the first to share one in the ‘List a home’ tab.")
    for listing in listings:
        with st.container(border=True):
            listing_photos = photos_by_listing.get(listing["id"], [])
            if listing_photos:
                st.image(listing_photos[0], use_container_width=True, caption="Cover photo")
                if len(listing_photos) > 1:
                    photo_columns = st.columns(min(4, len(listing_photos) - 1))
                    for photo_column, photo_data in zip(photo_columns, listing_photos[1:]):
                        photo_column.image(photo_data, use_container_width=True)
            else:
                st.markdown("<div style='padding:2.5rem 1rem;text-align:center;border-radius:14px;background:#eef2e9;color:#647368;font-size:1.1rem'>Ames home · photo coming soon</div>", unsafe_allow_html=True)
            title_col, price_col = st.columns([3, 1])
            title_col.markdown(f"### {listing['title']}")
            price_col.markdown(f"### ${listing['price']:,}")
            st.caption(f"{NEIGHBORHOODS.get(listing['neighborhood'], listing['neighborhood'])} · {listing['home_type']} · {listing['area']:,} sq ft · {listing['bedrooms']} bed · {listing['bathrooms']} bath · Built {listing['year_built']}")
            st.write(listing["description"])
            st.caption("Listed by a Ronald member · Contact through Ronald messages")
            if listing["phone"]:
                st.caption(f"Seller phone: {listing['phone']}")
            user = current_user()
            if user and user["id"] == listing["user_id"]:
                st.caption("This is your listing.")
            elif user:
                with st.form(f"contact_seller_{listing['id']}"):
                    message = st.text_area("Message the seller", placeholder="Hi, I’m interested in this home. Is it still available?", max_chars=2000, key=f"buyer_message_{listing['id']}")
                    sent = st.form_submit_button("Send message", type="primary")
                if sent:
                    if not message.strip():
                        st.error("Write a short message before sending.")
                    else:
                        with database() as db:
                            db.execute("INSERT INTO messages(listing_id, sender_id, recipient_id, body) VALUES (?, ?, ?, ?)", (listing["id"], user["id"], listing["user_id"], message.strip()))
                        st.success("Your message was sent. The seller can reply from Account & messages.")
            else:
                st.caption("Sign in or create an account in Account & messages to contact this seller.")

if current_page == "List a home":
    st.markdown('<div class="hero"><div class="eyebrow">SELLER SPACE</div><h1>Give your home a great first look.</h1><p>Add a few photos, share the details, and let interested buyers contact you through Ronald.</p></div>', unsafe_allow_html=True)
    st.subheader("Share a home for sale")
    user = current_user()
    if not user:
        st.info("Create a free account or sign in in the Account & messages tab before posting a home.")
    else:
        st.caption(f"Posting as {user['email']} · Your email is not publicly shown. Buyers can message you here.")
        with st.form("new_listing_form"):
            st.markdown("#### Start with your photos")
            st.caption("Your first photo becomes the cover image buyers see while browsing.")
            listing_photos = st.file_uploader("Property photos (optional)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, help="Add up to 5 photos. The first will be the cover. Each photo can be up to 5 MB.")
            title = st.text_input("Listing title", placeholder="Bright 3-bedroom home near College Creek", max_chars=100)
            description = st.text_area("Tell buyers about the home", placeholder="Share the details that make this home special…", max_chars=3000)
            c1, c2 = st.columns(2)
            with c1:
                list_home_type = st.selectbox("Home type", ["Detached house", "Two-storey house", "Townhome", "Duplex"], key="list_home_type")
                list_neighborhood_name = st.selectbox("Neighborhood", list(NEIGHBORHOODS.values()), key="list_neighborhood")
                list_price = st.number_input("Asking price ($)", min_value=1, max_value=10_000_000, value=250_000, step=5_000)
                list_area = st.number_input("Finished living area (sq ft)", min_value=200, max_value=20_000, value=1600, step=50)
            with c2:
                list_bedrooms = st.selectbox("Bedrooms", ["Studio", "1", "2", "3", "4", "5", "6+"])
                list_bathrooms = st.selectbox("Full bathrooms", ["1", "2", "3", "4", "5+"])
                list_year = st.number_input("Year built (if known)", min_value=1800, max_value=2026, value=2000, step=1)
                list_phone = st.text_input("Phone number (optional)", placeholder="Add a number buyers may call")
            submitted = st.form_submit_button("Publish listing", type="primary", use_container_width=True)
        if submitted:
            if not title.strip() or not description.strip():
                st.error("Add a listing title and a short description before publishing.")
            elif len(listing_photos) > 5:
                st.error("Choose up to 5 photos for one listing.")
            else:
                try:
                    cleaned_photos = [normalize_listing_photo(photo) for photo in listing_photos]
                    list_code = next(code for code, name in NEIGHBORHOODS.items() if name == list_neighborhood_name)
                    with database() as db:
                        cursor = db.execute("""INSERT INTO listings(user_id,title,description,home_type,neighborhood,price,area,bedrooms,bathrooms,year_built,phone)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (user["id"], title.strip(), description.strip(), list_home_type, list_code, int(list_price), int(list_area), list_bedrooms, list_bathrooms, int(list_year), list_phone.strip()))
                        listing_id = int(cursor.lastrowid)
                        db.executemany("INSERT INTO listing_photos(listing_id,image_data,position) VALUES (?,?,?)", [(listing_id, image_data, index) for index, image_data in enumerate(cleaned_photos)])
                    st.success("Your home is now listed on Ronald.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

if current_page == "Account":
    st.markdown('<div class="hero"><div class="eyebrow">YOUR RONALD SPACE</div><h1>Stay in touch.</h1><p>Sign in to message sellers, manage your listings, and keep up with conversations.</p></div>', unsafe_allow_html=True)
    st.subheader("Your Ronald account")
    user = current_user()
    if user is None:
        sign_in_tab, create_tab = st.tabs(["Sign in", "Create account"])
        with sign_in_tab:
            with st.form("sign_in_form"):
                login_email = st.text_input("Email address", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submit = st.form_submit_button("Sign in", type="primary")
            if login_submit:
                user_id = authenticate(login_email, login_password)
                if user_id:
                    st.session_state.ronald_user_id = user_id
                    st.rerun()
                st.error("We couldn’t sign you in with those details.")
        with create_tab:
            with st.form("create_account_form"):
                register_email = st.text_input("Email address", key="register_email")
                register_password = st.text_input("Create a password", type="password", key="register_password", help="Use at least 8 characters.")
                register_submit = st.form_submit_button("Create account", type="primary")
            st.caption("Email verification is not enabled yet. Your email is used for sign-in and is not shown on your listings.")
            if register_submit:
                try:
                    st.session_state.ronald_user_id = create_account(register_email, register_password)
                    st.success("Your account is ready.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("An account with that email already exists. Try signing in instead.")
                except ValueError as error:
                    st.error(str(error))
    else:
        account_col, logout_col = st.columns([4, 1])
        account_col.markdown(f"Signed in as **{user['email']}**")
        if logout_col.button("Sign out"):
            del st.session_state.ronald_user_id
            st.rerun()
        st.caption("Your email address hasn’t been verified. It stays private from buyers; use the optional phone field on a listing if you want to share a number.")
        received, sent_messages = st.tabs(["Inbox", "Sent"])
        for target_tab, direction in ((received, "received"), (sent_messages, "sent")):
            with target_tab:
                field = "recipient_id" if direction == "received" else "sender_id"
                with database() as db:
                    messages = db.execute(f"""SELECT messages.*, listings.title AS listing_title, sender.email AS sender_email, recipient.email AS recipient_email
                        FROM messages JOIN listings ON listings.id=messages.listing_id
                        JOIN users AS sender ON sender.id=messages.sender_id
                        JOIN users AS recipient ON recipient.id=messages.recipient_id
                        WHERE messages.{field}=? ORDER BY messages.created_at DESC, messages.id DESC""", (user["id"],)).fetchall()
                if not messages:
                    st.info("No messages here yet.")
                for msg in messages:
                    with st.container(border=True):
                        st.markdown(f"**{msg['listing_title']}** · {msg['created_at']}")
                        other_person = msg["sender_email"] if direction == "received" else msg["recipient_email"]
                        st.caption(("From " if direction == "received" else "To ") + other_person)
                        st.write(msg["body"])
                        if direction == "received":
                            with st.form(f"reply_message_{msg['id']}"):
                                reply = st.text_area("Reply", placeholder="Write a reply…", max_chars=2000, key=f"reply_body_{msg['id']}")
                                reply_sent = st.form_submit_button("Send reply")
                            if reply_sent:
                                if not reply.strip():
                                    st.error("Write a short reply before sending.")
                                else:
                                    with database() as db:
                                        db.execute("INSERT INTO messages(listing_id,sender_id,recipient_id,body) VALUES (?,?,?,?)", (msg["listing_id"], user["id"], msg["sender_id"], reply.strip()))
                                    st.success("Reply sent.")
                                    st.rerun()
        st.markdown("#### Your listings")
        with database() as db:
            my_listings = db.execute("SELECT id, title, price, neighborhood FROM listings WHERE user_id=? ORDER BY created_at DESC", (user["id"],)).fetchall()
        if not my_listings:
            st.caption("You haven’t posted a home yet.")
        for own_listing in my_listings:
            own_col, remove_col = st.columns([4, 1])
            own_col.write(f"**{own_listing['title']}** · ${own_listing['price']:,} · {NEIGHBORHOODS.get(own_listing['neighborhood'], own_listing['neighborhood'])}")
            if remove_col.button("Remove", key=f"remove_listing_{own_listing['id']}"):
                with database() as db:
                    db.execute("DELETE FROM listings WHERE id=? AND user_id=?", (own_listing["id"], user["id"]))
                st.rerun()
