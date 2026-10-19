# 🌱 Smart Irrigation System

A Streamlit-based educational dashboard that trains a Random Forest classifier from `smart_irrigation_system.csv` and predicts the dataset's `Irrigation_Need` class from user-provided feature values.

> **Important:** This is a learning/decision-support prototype. The model learns patterns in the supplied dataset. A high holdout score is not proof of real-world irrigation accuracy. Validate with agronomic expertise and independent field data before using for operational decisions.

## Features

- Dataset-driven categorical choices (no hard-coded label encoding for prediction inputs).
- Preprocessing pipeline with imputation and one-hot encoding.
- Random Forest classification model.
- Holdout accuracy, confusion matrix, and per-class precision/recall/F1.
- Interactive prediction form based on the dataset's actual feature columns.
- Dataset preview, class distribution, missing-value overview, and CSV download.

## Requirements

- Python 3.11 recommended
- `smart_irrigation_system.csv` in the same folder as `app.py`

## Run on Windows (PowerShell)

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Keep the terminal open while using the app. Press `Ctrl+C` in PowerShell to stop it.

## Dataset expectations

The CSV must include a target column named `Irrigation_Need`. All other columns are used as input features. The app discovers categorical and numeric feature columns from the CSV instead of assuming a fixed set of categories.

## Model and evaluation notes

- The app uses a fixed random seed for a repeatable 80/20 train/test split.
- Preprocessing is fitted inside a scikit-learn pipeline to keep training and prediction transformations consistent.
- The model is evaluated on the holdout split, then retrained on the full dataset for interactive predictions.
- The reported score is dataset-specific and should not be interpreted as a guarantee for unseen farms or future conditions.

## Project files

- `app.py` — Streamlit dashboard and model pipeline
- `smart_irrigation_system.csv` — dataset (must be present locally)
- `smart_irrigation_system.ipynb` — original notebook, if included
- `requirements.txt` — Python dependencies

## Credits

Retain and update attribution to the original project author/source as appropriate. If you redistribute or adapt this project, preserve any applicable license and acknowledgements from the original repository.
