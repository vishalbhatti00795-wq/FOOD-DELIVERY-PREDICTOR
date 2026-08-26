import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        border: 1px solid #e5e7eb;
    }

    .prediction-value {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
    }

    .prediction-label {
        color: #6b7280;
        font-size: 16px;
        margin-top: 5px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("food_delivery_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Create `food_delivery_model.pkl` "
        "using the model-saving code provided below."
    )
    st.stop()

# ---------------------------------------------------------
# Feature order used by the notebook
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Weather_Clear",
    "Weather_Foggy",
    "Weather_Rainy",
    "Weather_Snowy",
    "Weather_Windy",
    "Traffic_Level_High",
    "Traffic_Level_Low",
    "Traffic_Level_Medium",
    "Time_of_Day_Afternoon",
    "Time_of_Day_Evening",
    "Time_of_Day_Morning",
    "Time_of_Day_Night",
    "Vehicle_Type_Bike",
    "Vehicle_Type_Car",
    "Vehicle_Type_Scooter",
]

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<p class="main-title">🛵 Food Delivery Time Predictor</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Predict estimated food delivery time using a Linear Regression model.</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar - model information
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model", "Linear Regression")
    st.metric("R² Score", "0.820")
    st.metric("Adjusted R²", "0.816")

    st.divider()

    st.markdown("### Prediction Features")
    st.write("• Distance")
    st.write("• Preparation Time")
    st.write("• Courier Experience")
    st.write("• Weather")
    st.write("• Traffic Level")
    st.write("• Time of Day")
    st.write("• Vehicle Type")

    st.divider()
    st.caption("Built from the Food Delivery Time Predictor project.")

# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.subheader("📋 Delivery Information")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input(
        "Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=8.0,
        step=0.1,
        help="Distance between restaurant and customer."
    )

    preparation_time = st.number_input(
        "Preparation Time (minutes)",
        min_value=1,
        max_value=120,
        value=15,
        step=1,
        help="Approximate time required to prepare the order."
    )

    courier_experience = st.number_input(
        "Courier Experience (years)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        help="Experience of the delivery courier."
    )

with col2:
    weather = st.selectbox(
        "Weather",
        ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
    )

    traffic = st.selectbox(
        "Traffic Level",
        ["High", "Low", "Medium"]
    )

    time_of_day = st.selectbox(
        "Time of Day",
        ["Afternoon", "Evening", "Morning", "Night"]
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Bike", "Car", "Scooter"]
    )

# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------
st.divider()

predict_clicked = st.button("🚀 Predict Delivery Time")

if predict_clicked:

    # Start with numerical features
    input_data = {
        "Distance_km": distance,
        "Preparation_Time_min": preparation_time,
        "Courier_Experience_yrs": courier_experience,

        # Weather one-hot columns
        "Weather_Clear": 0,
        "Weather_Foggy": 0,
        "Weather_Rainy": 0,
        "Weather_Snowy": 0,
        "Weather_Windy": 0,

        # Traffic one-hot columns
        "Traffic_Level_High": 0,
        "Traffic_Level_Low": 0,
        "Traffic_Level_Medium": 0,

        # Time of day one-hot columns
        "Time_of_Day_Afternoon": 0,
        "Time_of_Day_Evening": 0,
        "Time_of_Day_Morning": 0,
        "Time_of_Day_Night": 0,

        # Vehicle one-hot columns
        "Vehicle_Type_Bike": 0,
        "Vehicle_Type_Car": 0,
        "Vehicle_Type_Scooter": 0,
    }

    # Set selected categorical values to 1
    input_data[f"Weather_{weather}"] = 1
    input_data[f"Traffic_Level_{traffic}"] = 1
    input_data[f"Time_of_Day_{time_of_day}"] = 1
    input_data[f"Vehicle_Type_{vehicle}"] = 1

    # Create dataframe in exact model feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_COLUMNS]

    # Prediction
    prediction = float(model.predict(input_df)[0])
    prediction = max(0, prediction)

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------
    st.subheader("🎯 Prediction")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Estimated Time", f"{prediction:.1f} min")

    with result_col2:
        st.metric("Distance", f"{distance:.1f} km")

    with result_col3:
        st.metric("Traffic", traffic)

    st.markdown(
        f"""
        <div class="prediction-box">
            <p class="prediction-value">{prediction:.1f} minutes</p>
            <p class="prediction-label">Estimated Food Delivery Time</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Human-friendly estimate
    hours = int(prediction // 60)
    minutes = int(round(prediction % 60))

    if hours > 0:
        readable_time = f"{hours} hr {minutes} min"
    else:
        readable_time = f"{minutes} min"

    st.success(f"Estimated delivery time: **{readable_time}**")

    # Show model input for transparency
    with st.expander("🔍 View Model Input"):
        st.dataframe(input_df, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("Food Delivery Time Predictor • Built with Python, Scikit-learn & Streamlit")

