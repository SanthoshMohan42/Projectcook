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
st.caption("Human-in-the-loop cooking recommendation system")

date_input = st.date_input("Select date")

# 🔹 FIXED TIME OPTIONS (AS REQUESTED)
TIME_OPTIONS = [
    "06:30","08:20","10:20","11:20","14:20","18:20","17:20",
    "17:30","18:30","08:45","09:45","14:30","10:00","12:45",
    "14:45","09:00","09:10","11:00","13:00","16:45","09:30",
    "11:30","14:00","15:00","11:15","15:30","12:15","13:30",
    "09:15","17:00","12:30","17:15"
]

time_input = st.selectbox("Batch time", TIME_OPTIONS)

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
# Encoding maps (UNCHANGED)
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
# Helper: Convert HH:MM → minutes
# --------------------------------------------------
def time_to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

# --------------------------------------------------
# Prediction logic (UNCHANGED CORE)
# --------------------------------------------------
if st.button("Predict chicken requirement"):
    date = pd.to_datetime(date_input)
    time_minutes = time_to_minutes(time_input)

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

    input_array = np.array(input_row, dtype=float).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    st.success(
        f"✅ Recommended chickens to cook: **{int(round(prediction))}**"
    )
