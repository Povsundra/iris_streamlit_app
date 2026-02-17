

import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

st.title("🌸 Iris Species Prediction")
st.write("Enter flower measurements, then click **Predict**.")

sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
sepal_width  = st.number_input("Sepal width (cm)",  min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
petal_width  = st.number_input("Petal width (cm)",  min_value=0.0, value=0.2, step=0.1)

if st.button("Predict"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.success(f"Prediction: **{target_names[pred]}**")
    st.write("Confidence:")
    st.bar_chart(proba)
