import streamlit as st
import pandas as pd
import joblib
import os

from utils import load_data

if not os.path.exists("history.csv"):
    df = load_data("train.zip")
    history_df = df[["show_date", "audience_count", "book_theater_id"]]
    history_df.to_csv("history.csv", index=False)

history_df = pd.read_csv("history.csv")
history_df["show_date"] = pd.to_datetime(history_df["show_date"])

model = joblib.load("best_model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("Audience Forecasting App")

date = st.date_input("Start Date")
days = st.slider("Days to Predict", 1, 7)
theater_id = st.text_input("Enter Theater ID")

def create_features(df):

    df = df.sort_values("show_date")

    df["day"] = df["show_date"].dt.day
    df["month"] = df["show_date"].dt.month
    df["year"] = df["show_date"].dt.year
    df["dayofweek"] = df["show_date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["weekofyear"] = df["show_date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["show_date"].dt.dayofyear
    df["week_of_month"] = (df["show_date"].dt.day - 1) // 7 + 1

    df["lag_1"] = df["audience_count"].shift(1)
    df["lag_3"] = df["audience_count"].shift(3)
    df["lag_7"] = df["audience_count"].shift(7)
    df["lag_14"] = df["audience_count"].shift(14)

    df["roll_mean_7"] = df["audience_count"].rolling(7).mean()
    df["roll_mean_14"] = df["audience_count"].rolling(14).mean()

    df["roll_std_7"] = df["audience_count"].rolling(7).std()
    df["roll_std_14"] = df["audience_count"].rolling(14).std()

    df["trend_7"] = df["roll_mean_7"] - df["lag_7"]
    df["trend_14"] = df["roll_mean_14"] - df["lag_14"]

    df["lag_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
    df["diff_1_3"] = df["lag_1"] - df["lag_3"]
    df["diff_7_14"] = df["lag_7"] - df["lag_14"]
    df["lag3_lag14_diff"] = df["lag_3"] - df["lag_14"]

    df["is_peak"] = (df["lag_1"] > df["roll_mean_7"] * 1.5).astype(int)

    df = df.fillna(method="ffill").fillna(method="bfill")

    return df

if st.button("Predict"):

    if theater_id == "":
        st.error("Enter Theater ID")
        st.stop()

    df = history_df.copy()
    predictions = []

    for i in range(days):

        future_date = pd.to_datetime(date) + pd.Timedelta(days=i)

        new_row = pd.DataFrame({
            "show_date": [future_date],
            "audience_count": [None],
            "book_theater_id": [theater_id]
        })

        df = pd.concat([df, new_row], ignore_index=True)

        df = create_features(df)

        input_row = df.iloc[-1:].drop(
            columns=["show_date", "audience_count"],
            errors="ignore"
        )

        input_row = input_row.reindex(columns=feature_order, fill_value=0)

        pred = model.predict(input_row)[0]

        df.loc[df.index[-1], "audience_count"] = pred

        predictions.append({
            "date": future_date,
            "prediction": pred
        })

    result_df = pd.DataFrame(predictions)

    st.write("Predictions:")
    st.dataframe(result_df)