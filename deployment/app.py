import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ── Load the saved model from Hugging Face model hub ──────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="anujgrover/tourism-project",
        filename="best_tourism_project_model_v1.joblib",
        repo_type="model",
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# ── Streamlit App UI ──────────────────────────────────────────────────────────
st.title("🌍 Tourism Package Purchase Prediction")
st.markdown("Predict whether a customer will purchase the **Wellness Tourism Package**.")

st.sidebar.header("Customer Details")

# Input features
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.sidebar.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.sidebar.number_input("Number of Trips", min_value=1, max_value=20, value=3)
passport = st.sidebar.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.sidebar.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.sidebar.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.sidebar.number_input("Monthly Income", min_value=10000, max_value=200000, value=30000)

st.sidebar.header("Customer Interaction Data")
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
number_of_followups = st.sidebar.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
duration_of_pitch = st.sidebar.number_input("Duration of Pitch (min)", min_value=1, max_value=60, value=15)

# ── Encode categorical features (same as training) ───────────────────────────
from sklearn.preprocessing import LabelEncoder

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch,
}])

# Encode categorical columns to match training
categorical_cols = input_data.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    input_data[col] = le.fit_transform(input_data[col])

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ The customer is **likely to purchase** the Wellness Tourism Package! (Probability: {probability[1]:.2%})")
    else:
        st.warning(f"❌ The customer is **unlikely to purchase** the package. (Probability: {probability[0]:.2%})")
