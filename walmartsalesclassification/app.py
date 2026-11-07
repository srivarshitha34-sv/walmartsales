import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Walmart Sales Classification", page_icon="🛒", layout="wide"
)

model = joblib.load("retail_model.joblib")
feature_names = joblib.load("retail_features.joblib")

st.title("🛒 Walmart Store Sales Classification Application")
st.write(
    "Predict whether weekly store performance will reach **High Sales** or"
    " **Low Sales**."
)

st.sidebar.header("Store & Economic Parameters")
store_id = st.sidebar.number_input(
    "Store ID", min_value=1, max_value=45, value=1
)
holiday_flag = st.sidebar.selectbox("Holiday Week? (1 = Yes, 0 = No)", [0, 1])
temperature = st.sidebar.number_input(
    "Temperature (°F)", min_value=10.0, max_value=110.0, value=70.0
)
fuel_price = st.sidebar.number_input(
    "Fuel Price ($)", min_value=1.0, max_value=5.0, value=3.5
)
cpi = st.sidebar.number_input("CPI (Consumer Price Index)", value=200.0)
unemployment = st.sidebar.number_input("Unemployment Rate", value=8.0)

input_values = pd.DataFrame([{
    "Store": store_id,
    "Holiday_Flag": holiday_flag,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
}])

input_df = pd.DataFrame(input_values, columns=feature_names)

st.subheader("Entered Store Parameters")
st.dataframe(input_df, use_container_width=True)

if st.button("Classify Sales Performance", type="primary"):
  prediction = model.predict(input_df)[0]
  st.subheader("Prediction Result")
  if prediction == 1:
    st.success(
        "Prediction: **High Sales Demand (Target Met)** 🚀.\n- Ensure full"
        " inventory stocking."
    )
  else:
    st.warning(
        "Prediction: **Low Sales Demand (Target Missed)** ⚠️.\n- Consider"
        " promotional markdowns."
    )