from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "machine_failure_model.joblib"
THRESHOLD = 0.4

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Manufacturing Quality Prediction",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Smart Manufacturing Quality Prediction")
st.write("Enter machine readings to estimate the risk of machine failure.")

with st.form("machine_data_form"):
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature (K)", 250.0, 350.0, 300.0)
    process_temp = st.number_input("Process temperature (K)", 250.0, 400.0, 310.0)
    rotational_speed = st.number_input(
        "Rotational speed (rpm)", 1, 10000, 1500
    )
    torque = st.number_input("Torque (Nm)", 0.1, 1000.0, 40.0)
    tool_wear = st.number_input("Tool wear (min)", 0, 1000, 100)

    submitted = st.form_submit_button("Predict Failure Risk")

if submitted:
    machine_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature difference [K]": process_temp - air_temp
    }])

    probability = float(model.predict_proba(machine_data)[0, 1])
    prediction = int(probability >= THRESHOLD)

    st.metric("Estimated failure probability", f"{probability:.2%}")

    if prediction:
        st.error("Failure risk detected")
    else:
        st.success("No failure predicted at the selected threshold.")

    st.caption(f"Classification threshold: {THRESHOLD:.0%}")
    st.progress(min(max(probability, 0.0), 1.0))
