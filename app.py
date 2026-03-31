import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.title("Audience Prediction App")

st.write("Enter input features below:")

# ================================
# BASIC FEATURES
# ================================
book_theater_id = st.number_input("Book Theater ID", step=1)
day = st.number_input("Day", min_value=1, max_value=31)
month = st.number_input("Month", min_value=1, max_value=12)
year = st.number_input("Year", min_value=2000, max_value=2100)
dayofweek = st.number_input("Day of Week (0=Mon)", min_value=0, max_value=6)
is_weekend = st.selectbox("Is Weekend", [0, 1])

weekofyear = st.number_input("Week of Year", min_value=1, max_value=52)
dayofyear = st.number_input("Day of Year", min_value=1, max_value=366)
week_of_month = st.number_input("Week of Month", min_value=1, max_value=5)

# ================================
# LAG FEATURES
# ================================
lag_1 = st.number_input("Lag 1")
lag_3 = st.number_input("Lag 3")
lag_7 = st.number_input("Lag 7")
lag_14 = st.number_input("Lag 14")

# ================================
# ROLLING FEATURES
# ================================
roll_mean_7 = st.number_input("Rolling Mean 7")
roll_mean_14 = st.number_input("Rolling Mean 14")

is_peak = st.selectbox("Is Peak", [0, 1])

roll_std_7 = st.number_input("Rolling Std 7")
roll_std_14 = st.number_input("Rolling Std 14")

trend_7 = st.number_input("Trend 7")
trend_14 = st.number_input("Trend 14")

# ================================
# DERIVED FEATURES
# ================================
lag_ratio_7 = st.number_input("Lag Ratio 7")
diff_1_3 = st.number_input("Diff 1-3")
diff_7_14 = st.number_input("Diff 7-14")
lag3_lag14_diff = st.number_input("Lag3 - Lag14 Diff")

# ================================
# CREATE DATAFRAME
# ================================
input_data = pd.DataFrame({
    "book_theater_id": [book_theater_id],
    "day": [day],
    "month": [month],
    "year": [year],
    "dayofweek": [dayofweek],
    "is_weekend": [is_weekend],
    "weekofyear": [weekofyear],
    "dayofyear": [dayofyear],
    "week_of_month": [week_of_month],

    "lag_1": [lag_1],
    "lag_3": [lag_3],
    "lag_7": [lag_7],
    "lag_14": [lag_14],

    "roll_mean_7": [roll_mean_7],
    "roll_mean_14": [roll_mean_14],

    "is_peak": [is_peak],

    "roll_std_7": [roll_std_7],
    "roll_std_14": [roll_std_14],

    "trend_7": [trend_7],
    "trend_14": [trend_14],

    "lag_ratio_7": [lag_ratio_7],
    "diff_1_3": [diff_1_3],
    "diff_7_14": [diff_7_14],
    "lag3_lag14_diff": [lag3_lag14_diff]
})

# ================================
# PREDICTION
# ================================
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Audience Count: {prediction[0]:.2f}")

