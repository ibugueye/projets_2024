import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt  # Assurez-vous d'importer pyplot pour la création de figures
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Charger les données
df = pd.read_csv('df_final.csv')  # Remplacez par le chemin correct de votre fichier de données
X = df.drop(columns=['TARGET'])  # Supposons que 'TARGET' est votre variable cible
y = df['TARGET']

# Filtrer pour obtenir uniquement les observations en défaut de paiement
defaulted_loans = df[df['TARGET'] == 1].reset_index(drop=True)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Formation du modèle
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Initialiser l'explainer SHAP
explainer = shap.Explainer(model, X_train)

# Interface Streamlit
st.title("Analyse de Défaut de Paiement avec SHAP")

# Sélection de l'observation en défaut de paiement
index_to_explain = st.slider("Sélectionnez l'indice de l'observation en défaut de paiement à expliquer", 0, len(defaulted_loans)-1, 0)
observation_to_explain = defaulted_loans.iloc[index_to_explain:index_to_explain+1].drop(columns=['TARGET'])

# Calcul des valeurs SHAP pour l'observation sélectionnée
shap_values = explainer(observation_to_explain)

# Afficher les valeurs SHAP pour l'observation sélectionnée en utilisant le contexte de figure explicitement
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)  # Limitez à 10 fonctionnalités pour la lisibilité
st.pyplot(fig)
