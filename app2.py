import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# URL of the Flask API
API_URL = 'http://127.0.0.1:5000/predict'  # Adjust this to the URL of your Flask API

# Interface utilisateur de l'application
st.title("Application de Prédiction de Défaut de Paiement")

# Define the structure (schema) of the input expected by your model with sliders
input_data = {
    'SK_ID_CURR': st.slider('SK_ID_CURR', min_value=100000, max_value=999999, value=100001, step=1),
    'CNT_CHILDREN': st.slider('CNT_CHILDREN', min_value=0, max_value=20, value=0, step=1),
    'AMT_INCOME_TOTAL': st.slider('AMT_INCOME_TOTAL', min_value=20000, max_value=1000000, value=100000, step=1000),
    'AMT_CREDIT': st.slider('AMT_CREDIT', min_value=50000, max_value=2000000, value=500000, step=10000),
    # Add other features as necessary
}

# Section for making prediction
if st.button('Prédire le risque de défaut'):
    # Send the data to the Flask API and get the prediction
    response = requests.post(API_URL, json=input_data)
    if response.status_code == 200:
        # Extract the prediction results
        result = response.json()
        probabilities = result['probabilities']
        prediction = result['prediction'][0]  # Assuming the first (and only) prediction
        
        # Display the results in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Résultat de la prédiction :")
            result_message = "Le client sera en défaut de paiement." if prediction == 1 else "Le client ne sera pas en défaut de paiement."
            st.write(result_message)

        with col2:
            st.subheader("Probabilités :")
            # Plotting the probability as a bar chart
            categories = ['Pas de défaut', 'Défaut']
            fig = px.bar(x=categories, y=probabilities, labels={'x': '', 'y': 'Probabilité'}, title="Probabilité de Défaut de Paiement")
            st.plotly_chart(fig)
    else:
        st.error('Erreur lors de la récupération de la prédiction')
