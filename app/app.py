import streamlit as st
from PIL import Image
import homepage, model1, model2, model3, model_ensemble
from hcdr.modeling.deep_learning import prediction

# Config
PAGES = {
    "Home": homepage,
    "Model XGBOOST" :model1,
    "Model HGBOOST" :model2,
    "Model CNN": model3,
    "model_ensemble": model_ensemble
}

# Sidebar


# st.sidebar.image(image, use_column_width=False, width=200)

st.cache()

X = pd.
y_pred = prediction(X, model_id=1)

selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]







