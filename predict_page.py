import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('housing_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data["model"]

def show_predict_page():
    st.title("Housing Price Prediction")
    st.write("""### Please provide the following details to get the housing price prediction""")

    # User input fields
    year_built = st.number_input("Year Built", min_value=1900, max_value=2023)
    square_footage = st.number_input("Square Footage", min_value=0)
    lot_sqft = st.number_input("Lot Square Footage", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathrooms = st.number_input("Bathrooms", min_value=0)
    days_to_sell = st.number_input("Days to Sell", min_value=0)

    # Dropdown for clusters
    cluster_options = [f"Cluster_{i}" for i in range(50)]
    selected_cluster = st.selectbox("Select a cluster", cluster_options)
    cluster_data = [1 if selected_cluster == option else 0 for option in cluster_options]

    ok = st.button("Calculate Housing Price")
    if ok:
        X = np.array([year_built, square_footage, lot_sqft, bedrooms, bathrooms, days_to_sell] + cluster_data)
        X = X.reshape(1, -1)
        price = regressor.predict(X)
        st.subheader(f"The estimated housing price is ${price[0]:.2f}")

show_predict_page()
