import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "linear_pipeline.joblib")

st.set_page_config(page_title="Housing Linear Model", layout="wide")

st.title("California Housing — Linear Regression Explorer")

@st.cache_data
def load_data(path="./housing.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.write(f"Loaded local dataset: {path}")
    else:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df.columns = list(data.feature_names) + ["MedHouseVal"]
        st.write("Fetched California housing from sklearn")
    return df

@st.cache_resource
def build_and_train_pipeline(df, target_name="median_house_value"):
    # detect target
    if target_name not in df.columns:
        # try common targets
        for cand in ["median_house_value", "MedHouseVal", "median_house_value_usd", "median_house_value"]:
            if cand in df.columns:
                target = cand
                break
        else:
            raise ValueError("Target column not found")
    else:
        target = target_name

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object, "category"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    pipeline = Pipeline([("preprocessor", preprocessor), ("lr", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # Persist model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, rmse


def get_feature_importance(pipeline, X_sample):
    # Extract feature names and coefficients when possible
    pre = pipeline.named_steps["preprocessor"]
    lr = pipeline.named_steps["lr"]
    try:
        num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_sample.select_dtypes(include=[object, "category"]).columns.tolist()
        # get feature names
        num_features = num_cols
        cat_features = []
        if cat_cols:
            cat_features = pre.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols).tolist()
        feature_names = [f"num__{c}" for c in num_features] + [f"cat__{c}" for c in cat_features]
        coefs = lr.coef_
        df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        df["abs_coef"] = df["coef"].abs()
        df = df.sort_values("abs_coef", ascending=False)
        return df
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")
        return None


# App layout
with st.sidebar:
    st.header("Data & Model")
    if st.button("(Re)train model"):
        df = load_data()
        with st.spinner("Training..."):
            pipeline, rmse = build_and_train_pipeline(df)
        st.success(f"Trained. RMSE on hold-out: {rmse:.2f}")
    if st.button("Load persisted model"):
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
            st.success("Model loaded from disk")
        else:
            st.error("No persisted model found. Train first.")

    uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"]) 
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.write("Uploaded sample:")
            st.write(df_up.head())
            if st.button("Predict uploaded CSV"):
                if os.path.exists(MODEL_PATH):
                    pipeline = joblib.load(MODEL_PATH)
                    preds = pipeline.predict(df_up)
                    df_up["prediction"] = preds
                    st.download_button("Download predictions CSV", df_up.to_csv(index=False).encode("utf-8"), file_name="preds.csv")
                else:
                    st.error("Train or load a model first")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# Main area
st.subheader("Dataset")
try:
    df = load_data()
    st.dataframe(df.head())
    if st.button("Train and show metrics"):
        pipeline, rmse = build_and_train_pipeline(df)
        st.metric("Hold-out RMSE", f"{rmse:.2f}")
        st.success("Model trained and saved to models/linear_pipeline.joblib")

    # Show feature importances if model exists
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        st.subheader("Top coefficients")
        df_imp = get_feature_importance(pipeline, df.drop(columns=[c for c in df.columns if c.lower().startswith("median_house") or c.lower().startswith("medhouse")], errors='ignore'))
        if df_imp is not None:
            st.table(df_imp.head(20).assign(coef=lambda d: d.coef.round(2)))

except Exception as e:
    st.error(f"Failed to initialize app: {e}")

st.info("Tip: Use the sidebar to train, load, or upload data for predictions.")
