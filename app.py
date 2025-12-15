import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Project COOK – PDT Recommendation",
    layout="centered"
)

# --------------------------------------------------
# FEATURE ORDER (MUST MATCH TRAINING EXACTLY)
# --------------------------------------------------
FEATURE_COLUMNS = [
    "Human_Traffic",
    "Weather",
    "Public_Event",
    "Shredded_chicken",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "Time_minutes"
]

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("pdt_recommendation_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍗 Project COOK – Smart PDT Recommendation")
st.caption("Human-in-the-loop chicken production recommendation")

date_input = st.date_input("Select date")
time_input = st.time_input("Batch time (HH:MM)")

shredded = st.number_input(
    "Shredded chicken prepared (units)",
    min_value=0,
    value=12
)

human_traffic = st.selectbox(
    "Expected customer demand",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

weather = st.selectbox(
    "Weather condition",
    ["Cold", "Warm", "Hot", "Rainy"]
)

public_event = st.selectbox(
    "Public / Store Event",
    ["No", "Yes"]
)

# --------------------------------------------------
# Encoding maps (MUST MATCH TRAINING)
# --------------------------------------------------
traffic_map = {
    "Much Lower": -2,
    "Neutral": 0,
    "Higher": 1,
    "Much Higher": 2
}

weather_map = {
    "Cold": 1,
    "Warm": 0,
    "Hot": -1,
    "Rainy": 2
}

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict chicken requirement"):
    date = pd.to_datetime(date_input)

    # Convert time → minutes since midnight
    time_minutes = time_input.hour * 60 + time_input.minute

    input_row = [
        traffic_map[human_traffic],
        weather_map[weather],
        1 if public_event == "Yes" else 0,
        shredded,
        date.dayofweek,
        date.day,
        int(date.isocalendar().week),
        time_minutes
    ]

    # Convert to NumPy array (shape = (1, 8))
    input_array = np.array(input_row, dtype=float).reshape(1, -1)

    # Safety debug (remove later)
    st.write("Model input (ordered):", dict(zip(FEATURE_COLUMNS, input_row)))
    st.write("Input shape:", input_array.shape)

    prediction = model.predict(input_array)[0]

    st.success(
        f"✅ Recommended chickens to cook: **{int(round(prediction))}**"
    )
