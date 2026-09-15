from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "instagram_engagement_processed.csv"

df = pd.read_csv(DATA_PATH)

st.title("Instagram Engagement Prediction")
st.dataframe(df.head(10))
