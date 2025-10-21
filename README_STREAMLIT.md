# Streamlit app for California Housing Linear Regression

This repository includes a small Streamlit app (`streamlit_app.py`) that trains a LinearRegression model on the California housing dataset. It will use `./housing.csv` if present in the working directory; otherwise it will fetch the dataset from scikit-learn.

Requirements

- Python 3.8+

Install dependencies (recommended in a virtual environment):

PowerShell commands:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-streamlit.txt
```

Run the app:

```powershell
streamlit run streamlit_app.py
```

App features

- Train a LinearRegression pipeline (ColumnTransformer for numeric/categorical preprocessing)
- Persist the trained pipeline to `models/linear_pipeline.joblib`
- Upload a CSV and get predictions (downloadable CSV)
- Inspect top coefficients interactively

Notes

- The app uses caching to avoid retraining on every rerun. If you change code, use Streamlit's "R" or restart the app to clear caches.
- If your `housing.csv` has a different target column name, modify the `load_data` / `build_and_train_pipeline` functions in `streamlit_app.py` accordingly.
