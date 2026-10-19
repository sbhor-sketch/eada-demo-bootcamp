# Ronald · Ames Home Portal

Ronald is a local Streamlit website for home-price estimates and community listings in Ames, Iowa. It uses the Ames Housing training data in `Data/archive/train.csv` and the neural-network approach with the lowest validation RMSE in `house_price_model_report.json`. The Ronald house logo is in `ronald-logo.svg`.

## Run it

Install the Python packages once, then start Ronald:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

The project config binds Streamlit to `127.0.0.1`, so this setup is reachable only from the same computer. To let buyers use Ronald from the internet later, it will need to be deliberately deployed to a hosting service.

The Streamlit app opens on a welcome page with illustrated home styles and links into separate Estimate, Explore homes, List a home, and Account pages. Sellers can upload up to five photos; the first is shown as the listing cover. Each estimate includes a year-by-year chart and table based on the Ames FHFA house-price index. Listings, accounts, photos, and messages are stored in the local `ronald.sqlite3` SQLite database, created automatically the first time the app runs. Passwords are stored as salted PBKDF2 hashes. Email addresses are not verified yet. Seller email addresses are not shown on listings; a seller can optionally share a phone number.

`portal.html` is a standalone, offline-friendly price estimator only. It does not include accounts, listing submission, or messaging; those features need the running Streamlit app and its SQLite database.

To regenerate the browser model after changing the training data or model, install the project dependencies and run:

```bash
python -m pip install -r requirements.txt
python export_browser_model.py
```

`serve_portal.py` remains available as a legacy local estimator server and does not provide the SQLite marketplace features.

## Scope

The source dataset describes historical sales in Ames, Iowa and covers houses, townhomes, and duplexes. It does not contain apartment-specific records or geographic neighborhood boundaries. The neighborhood selector uses the dataset's neighborhood names; the estimate is educational and is not a formal appraisal.

The portal also shows a rough 2026 market-adjusted reference using the FHFA Ames MSA all-transactions house price index from FRED (Q2 2008 to Q2 2026). This is a broad market trend adjustment, not a model retrained on recent property sales. It cannot capture a specific home's appreciation or neighborhood-level changes, and its added uncertainty is not measured.

## Local marketplace note

Ronald is a simple local prototype, not a public production service. The SQLite database is stored beside the app and should be kept private. Email addresses are not confirmed, and the app does not yet include password reset, account recovery, moderation, or production-grade abuse protection. Do not expose the development server to the public internet with real user data.
