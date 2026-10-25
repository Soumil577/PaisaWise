# PaisaWise — Privacy-Minded UPI Expense Analytics

PaisaWise is a local-first expense tracker with a browser dashboard, budget analysis, recurring-payment hints, unusual-transaction alerts and ML-assisted expense categorisation.

## What changed in this version

- **Primary training data:** `data/upi_transactions_2024.csv` (the CSV supplied for this project, 250,000 rows).
- **Structured ML model:** a scikit-learn logistic-regression pipeline trained from transaction type, amount, transaction status, day of week, hour, weekend flag, device type and network type.
- **No synthetic generator in the runtime project:** the old synthetic text generator and its model have been removed. The training script does not generate fake rows.
- **Description-only workflow preserved:** the existing `/api/classify` route still accepts a description and amount. Since the supplied CSV has no narration/description column, this route now uses transparent keyword rules plus exact-match user corrections. It does not pretend the structured model learned text descriptions.
- **Structured model endpoint:** `POST /api/classify-structured` accepts the structured fields used during training and returns a category, confidence and review flag.
- **Metrics:** `model/metrics.json` reports a stratified 80/20 hold-out evaluation. The score is intentionally reported as measured; do not interpret it as guaranteed accuracy on other banks or users.

## Run on Windows

1. Extract the ZIP.
2. Install Python 3.12 (64-bit) from https://www.python.org/downloads/release/python-31210/ and enable **Add Python to PATH**.
3. Open the extracted `paisawise` folder, click the File Explorer address bar, type `cmd`, and press Enter.
4. Run:

   ```bat
   run.bat
   ```

5. Wait for the Flask server to start, then open http://127.0.0.1:8000.
6. Keep the terminal open while using the app. Press `Ctrl+C` in that terminal to stop it.

The ZIP includes the trained model, so a normal launch does not retrain it. The launcher installs the packages in `requirements.txt` into a local `.venv` environment.

## Retrain the structured model

From the project root, with the virtual environment active:

```bat
python backend\train_model.py
```

By default, this reads `data/upi_transactions_2024.csv`, creates a reproducible stratified 80/20 split, evaluates a logistic-regression baseline, then saves:

- `model/structured_category_model.joblib`
- `model/metrics.json`

To use another labelled CSV:

```bat
python backend\train_model.py --csv path\to\your_transactions.csv
```

The CSV must include a category/merchant_category column, an amount column, and a transaction-type column. Optional time, status, device and network fields are derived or filled as unknown when absent.

## API overview

- `GET /api/health` — model source, classes, training size and hold-out metrics.
- `POST /api/classify` — `{ "items": [{"description":"Swiggy dinner", "amount":350}] }`; keyword categorisation plus saved corrections.
- `POST /api/classify-structured` — `{ "items": [{"transaction_type":"P2M", "amount":350, "timestamp":"2024-10-08T12:30:00", "transaction_status":"SUCCESS", "device_type":"Android", "network_type":"4G"}] }`; structured ML prediction.
- `POST /api/feedback` — stores a user's exact-description category correction locally in `data/corrections.json`.
- `POST /api/analyze` — spending analysis for transaction history submitted by the browser.

## Dataset fields and category mapping

The training script maps the supplied dataset's `merchant_category` to the app's labels:

- `Grocery` → `Food`
- `Fuel` → `Transport`
- `Utilities` → `Bills`
- Other matching categories retain their corresponding app label.

The model deliberately excludes transaction IDs, sender/receiver age groups, sender state, bank names and the fraud flag. These fields are not needed for the intended category task and could introduce privacy or target-leakage concerns.

## Important limitations

1. The supplied CSV has no transaction narration or free-text description. Therefore it cannot directly train the description NLP model. Free-text classification uses transparent keyword rules until a labelled narration dataset is available.
2. The dataset's real-world provenance/authenticity has not been independently verified by this code. The app describes it as the supplied training CSV, not as independently verified genuine bank data.
3. The measured hold-out performance is modest. The supplied fields may have weak relationships with expense category. Do not report a high accuracy claim; use the values in `model/metrics.json` and validate against properly labelled real transactions before making deployment claims.
4. The 80/20 split is random and comes from the same source dataset. It is not a test of generalisation to unseen banks, populations or time periods.
5. The project is not yet an automatic SMS/UPI listener. Users can enter or import transactions, and automatic ingestion would need a separate Android component and explicit permissions.
6. Browser transaction records live in IndexedDB. If the backend is deployed remotely, API requests send transaction information to that server; use a local server for local-first testing.

## Tests

```bat
pip install -r requirements-dev.txt
python -m pytest backend/tests
```

The API tests cover health, description-rule categorisation, user corrections, structured model inference, analytics, validation and security headers.
