import streamlit as st
import requests
import pandas as pd

# URL of the Flask API
API_URL = 'http://127.0.0.1:5000/predict'  # Adjust this to the URL of your Flask API

# Interface utilisateur de l'application
st.title("Application de Prédiction de Défaut de Paiement")

# Manually define the structure (schema) of the input expected by your model
# This is a simplified example; you might need to collect more details based on your model's requirements
input_data = {
    'SK_ID_CURR': st.number_input('SK_ID_CURR', min_value=100000, max_value=999999, value=100001),
    'CNT_CHILDREN': st.number_input('CNT_CHILDREN', min_value=0, max_value=20, value=0),
    'AMT_INCOME_TOTAL': st.number_input('AMT_INCOME_TOTAL', min_value=20000, value=100000),
    'AMT_CREDIT': st.number_input('AMT_CREDIT', min_value=0, value=500000),
   
}

 
if st.button('Prédire le risque de défaut'):
    response = requests.post(API_URL, json=input_data)
    if response.status_code == 200:
        result = response.json()
        probabilities = result['probabilities']
        prediction = result['prediction'][0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Résultat de la prédiction :")
            result_message = "Défaut" if prediction == 1 else "Pas de défaut"
            st.write(result_message)

            st.subheader("Probabilités :")
            categories = ['Pas de défaut', 'Défaut']
            fig = px.bar(x=categories, y=probabilities, labels={'x': '', 'y': 'Probabilité'}, title="Probabilité de Défaut de Paiement")
            st.plotly_chart(fig)

        with col2:
            st.write("Interprétation globale avec SHAP")
            # SHAP Summary Plot...
            # Notez que vous devrez ajuster ou supprimer ces blocs selon la logique de votre application
            # Car ils nécessitent que le modèle et les shap_values soient définis

        with col3:
            st.write("Interprétation locale avec SHAP")
            # SHAP Waterfall Plot...
            # Ajustez ou supprimez également en fonction de la disponibilité de votre modèle et des données

 

    # Ici, vous pouvez ajouter du contenu supplémentaire sous les prédictions
            st.header("Détails et Informations Supplémentaires")
            st.write("Ici, vous pouvez ajouter des informations supplémentaires pertinentes pour l'utilisateur. Par exemple :")
            st.write("- Des conseils sur les prochaines étapes à suivre selon le résultat de la prédiction.")
            st.write("- Des liens vers des ressources externes ou des articles pour en savoir plus.")
            st.write("- Des explications détaillées sur comment les prédictions sont générées ou sur la signification des différentes probabilités.")

    else:   
        st.error('Erreur lors de la récupération de la prédiction')

