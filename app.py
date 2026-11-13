import streamlit as st
import pandas as pd
from ml_engine import predict_price
from llm_analyst import generate_market_report

st.set_page_config(
    page_title="California Housing AI Analyst",
    layout="wide"
)

st.title("California Housing Valuation & AI Analyst")
st.markdown(
    "A hybrid Machine Learning & Generative AI pipeline. The **Random Forest model** "
    "estimates property valuation based on historical census metrics, while **GPT-4o-mini** "
    "acts as an investment consultant delivering market insights."
)

st.sidebar.header("Property & Area Specifications")

med_inc = st.sidebar.slider("Median Income (tens of thousands $)", min_value=0.5, max_value=15.0, value=8.32, step=0.1)
house_age = st.sidebar.slider("House Age (Years)", min_value=1.0, max_value=52.0, value=41.0, step=1.0)
ave_rooms = st.sidebar.slider("Average Rooms", min_value=1.0, max_value=15.0, value=6.98, step=0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.02, step=0.05)
population = st.sidebar.slider("Block Population", min_value=5.0, max_value=5000.0, value=322.0, step=10.0)
ave_occup = st.sidebar.slider("Average Occupancy", min_value=1.0, max_value=8.0, value=2.55, step=0.1)

st.sidebar.subheader("Geographic Coordinates")
latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, step=0.01)
longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.23, step=0.01)

property_data = {
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude
}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Selected Features")
    features_df = pd.DataFrame(list(property_data.items()), columns=["Metric", "Value"])
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    analyze_button = st.button("Estimate Value & Generate Report", use_container_width=True, type="primary")

with col2:
    if analyze_button:
        input_df = pd.DataFrame([property_data])

        predicted_array = predict_price(input_df)
        estimated_price = float(predicted_array[0])

        st.subheader("Valuation Result")
        st.metric(
            label="Random Forest Estimated Value",
            value=f"${estimated_price:,.2f}"
        )
        st.divider()

        with st.spinner("AI Analyst is evaluating area data and generating the report..."):
            try:
                report = generate_market_report(property_data, estimated_price)
                st.subheader("AI Investment Analysis Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"Error communicating with OpenAI API: {e}")
    else:
        st.info("Adjust parameters from the sidebar and click the button to generate valuation and report.")