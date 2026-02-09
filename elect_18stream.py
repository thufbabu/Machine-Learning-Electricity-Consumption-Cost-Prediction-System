import streamlit as st
import pandas as pd
import pickle


# Load saved models & encoders

lr = pickle.load(open("lr_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
ohe1 = pickle.load(open("Appliance_Type.sav", "rb"))
le = pickle.load(open("Meter_Type.sav", "rb"))
le1 = pickle.load(open("Peak_Hour_Usage.sav", "rb"))

st.set_page_config(page_title="Electricity Cost Prediction", layout="centered")
st.title("⚡ Electricity Cost Prediction System")

st.markdown("Enter the details below to predict electricity cost")

# USER INPUTS

site_area = st.number_input("Site Area", min_value=0.0)
water_consumption = st.number_input("Water Consumption", min_value=0.0)
recycling_rate = st.number_input("Recycling Rate", min_value=0.0)
utilisation_rate = st.number_input("Utilisation Rate", min_value=0.0)
air_quality_index = st.number_input("Air Quality Index", min_value=0.0)
issue_resolution_time = st.number_input("Issue Resolution Time", min_value=0.0)
resident_count = st.number_input("Resident Count", min_value=0, step=1)

structure_type = st.selectbox(
    "Structure Type",
    ["Mixed-use", "Residential", "Commercial", "Industrial"]
)

appliance_type = st.selectbox(
    "Appliance Type",
    ["AC", "Fan", "Heater", "Refrigerator", "Washing Machine"]
)

meter_type = st.selectbox(
    "Meter Type",
    ["Smart", "Analog"]
)

usage_pattern = st.selectbox(
    "Usage Pattern",
    ["High", "Medium", "Low"]
)

peak_hour_usage = st.selectbox(
    "Peak Hour Usage",
    ["Yes", "No"]
)

weather_condition = st.selectbox(
    "Weather Condition",
    ["Rainy", "Winter", "Summer"]
)


# PREDICTION BUTTON

if st.button("Predict Electricity Cost"):
    # Create input DataFrame
    data = pd.DataFrame([{
        'site area': site_area,
        'structure type': structure_type,
        'water consumption': water_consumption,
        'recycling rate': recycling_rate,
        'utilisation rate': utilisation_rate,
        'air qality index': air_quality_index,
        'issue reolution time': issue_resolution_time,
        'resident count': resident_count,
        'Appliance_Type': appliance_type,
        'Meter_Type': meter_type,
        'Usage_Pattern': usage_pattern,
        'Peak_Hour_Usage': peak_hour_usage,
        'Weather_Condition': weather_condition
    }])

   
    # Encoding (same as training)
  
    data['Usage_Pattern'] = data['Usage_Pattern'].replace({
        'High': 0, 'Medium': 1, 'Low': 2
    }).infer_objects(copy=False)

    data['structure type'] = data['structure type'].replace({
        'Mixed-use': 0,
        'Residential': 1,
        'Commercial': 2,
        'Industrial': 3
    }).infer_objects(copy=False)

    data['Weather_Condition'] = data['Weather_Condition'].replace({
        'Rainy': 0, 'Winter': 1, 'Summer': 2
    }).infer_objects(copy=False)

    data['Meter_Type'] = le.transform(data['Meter_Type'])
    data['Peak_Hour_Usage'] = le1.transform(data['Peak_Hour_Usage'])

    # One-Hot Encoding
    appliance_encoded = ohe1.transform(data[['Appliance_Type']])
    appliance_df = pd.DataFrame(
        appliance_encoded,
        columns=ohe1.get_feature_names_out(),
        index=data.index
    )

    data = pd.concat([data.drop('Appliance_Type', axis=1), appliance_df], axis=1)

    # Ensure column order
    data = data[scaler.feature_names_in_]

    # Scaling & Prediction
    data_scaled = scaler.transform(data)
    prediction = lr.predict(data_scaled)

    st.success(f"💡 Predicted Electricity Cost: ₹ {round(prediction[0], 2)}")
