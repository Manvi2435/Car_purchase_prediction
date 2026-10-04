import streamlit as st
import joblib
import pandas as pd


# -----------------------------
# Load model and preprocessor
# -----------------------------
model = joblib.load("models/final_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Car Purchase Prediction",
    page_icon="🚗",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Purchase Prediction")
st.write(
    "Enter the details of a used car to predict whether "
    "the model considers it a potential purchase."
)


# -----------------------------
# Load dataset for dropdowns
# -----------------------------
df = pd.read_csv("data/car predict.csv")


# -----------------------------
# Input section
# -----------------------------
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    car_company = st.selectbox(
        "Car Company",
        sorted(df["Car_Company"].unique())
    )

    available_models = sorted(
        df.loc[
            df["Car_Company"] == car_company,
            "Model"
        ].unique()
    )

    car_model = st.selectbox(
        "Model",
        available_models
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=2022
    )

    km_driven = st.selectbox(
        "KM Driven",
        [
            "0-20k",
            "20k-50k",
            "50k-80k",
            "80k-120k",
            "120k+"
        ]
    )

    engine_cc = st.selectbox(
        "Engine Capacity",
        [
            "<1000cc",
            "1000-1500cc",
            "1500-2000cc",
            "2000cc+"
        ]
    )

with col2:

    mileage = st.selectbox(
        "Mileage",
        [
            "Low (<12 km/l)",
            "Average (12-18 km/l)",
            "Good (18-24 km/l)",
            "Excellent (24+ km/l)"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        sorted(df["Transmission"].unique())
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(df["Fuel_Type"].unique())
    )

    owner_type = st.selectbox(
        "Owner Type",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above"
        ]
    )

    insurance = st.selectbox(
        "Insurance Valid",
        sorted(df["Insurance_Valid"].unique())
    )

    accident = st.selectbox(
        "Accident History",
        sorted(df["Accident_History"].unique())
    )


# Prediction button

if st.button(" Predict Purchase Decision"):

    user_data = pd.DataFrame({
        "Car_Company": [car_company],
        "Model": [car_model],
        "Year": [year],
        "KM_Driven": [km_driven],
        "Engine_CC": [engine_cc],
        "Mileage": [mileage],
        "Transmission": [transmission],
        "Fuel_Type": [fuel_type],
        "Owner_Type": [owner_type],
        "Insurance_Valid": [insurance],
        "Accident_History": [accident]
    })

    user_processed = preprocessor.transform(user_data)

    prediction = model.predict(user_processed)[0]

    yes_index = list(model.classes_).index("Yes")

    buy_probability = model.predict_proba(
        user_processed
    )[0][yes_index]

    # -----------------------------
    # Model Explanation
    # -----------------------------

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    # Convert processed input to a normal array
    user_array = user_processed.toarray().flatten()

    # Calculate contribution of each feature
    contributions = user_array * coefficients

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    # Sort by strongest contribution
    explanation["Absolute_Contribution"] = (
        explanation["Contribution"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute_Contribution",
        ascending=False
    )

    # Make feature names user-friendly
    def clean_feature_name(feature):

        feature = feature.replace("cat__", "")
        feature = feature.replace("ord__", "")
        feature = feature.replace("num__", "")

        # Categorical features
        feature = feature.replace(
            "Insurance_Valid_Yes",
            "Insurance Valid = Yes"
        )
        feature = feature.replace(
            "Insurance_Valid_No",
            "Insurance Valid = No"
        )

        feature = feature.replace(
            "Accident_History_Yes",
            "Accident History = Yes"
        )
        feature = feature.replace(
            "Accident_History_No",
            "Accident History = No"
        )

        feature = feature.replace(
            "Car_Company_",
            "Car Company = "
        )
        feature = feature.replace(
            "Model_",
            "Model = "
        )
        feature = feature.replace(
            "Transmission_",
            "Transmission = "
        )
        feature = feature.replace(
            "Fuel_Type_",
            "Fuel Type = "
        )

        # Ordinal features
        if feature == "KM_Driven":
            return f"KM Driven = {km_driven}"

        if feature == "Engine_CC":
            return f"Engine Capacity = {engine_cc}"

        if feature == "Mileage":
            return f"Mileage = {mileage}"

        if feature == "Owner_Type":
            return f"Owner Type = {owner_type}"

        # Numerical feature
        if feature == "Year":
            return f"Vehicle Year = {year}"

        return feature


    explanation["Feature"] = explanation["Feature"].apply(
        clean_feature_name
    )
    

    st.divider()

    st.subheader("Prediction")

    st.metric(
    label="Purchase Probability",
    value=f"{buy_probability:.1%}"
    )

    st.progress(float(buy_probability))

    # Probability interpretation
    if buy_probability >= 0.75:
        st.info("🟢 High purchase probability")
    elif buy_probability >= 0.50:
        st.info("🟡 Moderate purchase probability")
    else:
        st.info("🔴 Low purchase probability")

    if prediction == "Yes":
        st.success(
            "### Purchase Decision: YES ✅"
        )
    else:
        st.error(
            "### Purchase Decision: NO ❌"
        )

    # -----------------------------
    # Why did the model predict this?
    # -----------------------------

    st.subheader("🔎 Why did the model make this prediction?")

    st.write(
        "The features below had the strongest influence on the "
        "model's prediction. Positive values contributed toward "
        "the prediction, while negative values contributed against it."
    )

    top_explanations = explanation.head(5).copy()

    # Create readable impact labels
    top_explanations["Impact"] = top_explanations["Contribution"].apply(
        lambda x: "Positive" if x > 0 else "Negative"
    )

    top_explanations["Strength"] = top_explanations[
        "Contribution"
    ].abs().round(2)

    # Display explanation table
    st.dataframe(
        top_explanations[
            ["Feature", "Contribution", "Impact", "Strength"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Contribution chart
    st.write("### 📊 Contribution Strength")

    chart_data = top_explanations[
        ["Feature", "Contribution"]
    ].set_index("Feature")

    st.bar_chart(chart_data)