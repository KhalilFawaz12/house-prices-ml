import streamlit as st
import pandas as pd
import os
from src.predict import load_model, predict_raw_df, MODEL_PATH
# ensure project root (the folder containing this app.py) is on sys.path
import sys
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction - Demo")

model_path = os.path.join("results", "final_model.pkl")
if not os.path.exists(model_path):
    st.error("Model not found in results/final_model.pkl. Save the model from your final notebook first.")
else:
    model = load_model(model_path)

    st.markdown("Upload a CSV with the same raw columns as `data/train.csv` (unprocessed).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Uploaded preview:")
        st.dataframe(df.head())
        if st.button("Predict"):
            preds = predict_raw_df(df, model)
            out = pd.DataFrame({"Id": df.get("Id", pd.RangeIndex(start=1, stop=len(df)+1)), "SalePrice": preds})
            st.write("Predictions")
            st.dataframe(out.head(10))
            st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
    else:
        st.info("No file uploaded. Use a sample row from your train.csv for a quick demo.")
        if st.button("Load sample row from train.csv"):
            train_path = os.path.join("data", "train.csv")
            if os.path.exists(train_path):
                sample = pd.read_csv(train_path).drop(columns=["SalePrice"], errors="ignore").iloc[[0]]
                st.write("Sample input (one row):")
                st.dataframe(sample)
                preds = predict_raw_df(sample, model)
                st.write("Prediction:", preds[0])
            else:
                st.error("data/train.csv not found. Place train.csv in data/ or upload a CSV above.")

