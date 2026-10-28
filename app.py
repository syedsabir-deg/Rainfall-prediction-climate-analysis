
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Page configuration
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

# 2. Load the saved model and dataset
@st.cache_resource
def load_model():
    return joblib.load("Models/rainfall_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Data/Processed/rainfall_combined.csv")

model = load_model()
df = load_data()

# 3. Application heading
st.title("🌧️ Rainfall Prediction and Seasonal Climate Analysis")
st.write(
    "Predict monthly rainfall using historical district, year, "
    "and month data."
)

st.info(
    "This application provides estimates from historical patterns. "
    "It is not a real-time weather forecast."
)

# 4. User inputs
st.header("Predict Monthly Rainfall")

districts = sorted(df["district"].unique().tolist())

with st.form("prediction_form"):
    district = st.selectbox("Select district ID", districts)

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Year",
            min_value=1981,
            max_value=2100,
            value=2025,
            step=1
        )

    with col2:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(
                year=2000, month=m, day=1
            ).strftime("%B")
        )

    submitted = st.form_submit_button("Predict Rainfall")

if submitted:
    input_data = pd.DataFrame({
        "district": [district],
        "year": [int(year)],
        "month": [int(month)]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric(
        label="Estimated monthly rainfall (dataset units)",
        value=f"{max(0, prediction):.3f}"
    )

    st.caption(
        "The displayed estimate is prevented from going below zero. "
        "Model accuracy may be lower for years outside its training period."
    )

# 5. Seasonal rainfall analysis
st.divider()
st.header("Seasonal Rainfall Patterns")

st.write(
    "Explore average rainfall by month across all districts "
    "and years in the dataset."
)

monthly_rainfall = (
    df.groupby("month")["rainfall"]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    pd.Timestamp(year=2000, month=m, day=1).strftime("%b")
    for m in range(1, 13)
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(month_names, monthly_rainfall.values)
ax.set_xlabel("Month")
ax.set_ylabel("Average rainfall (dataset units)")
ax.set_title("Average Monthly Rainfall Across All Districts")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

wettest_month = int(monthly_rainfall.idxmax())
driest_month = int(monthly_rainfall.idxmin())

st.write(
    f"**Wettest month in the dataset:** "
    f"{month_names[wettest_month - 1]}"
)

st.write(
    f"**Driest month in the dataset:** "
    f"{month_names[driest_month - 1]}"
)

# 6. Show a sample of the dataset
with st.expander("View sample dataset records"):
    st.dataframe(df.head(10), use_container_width=True)
