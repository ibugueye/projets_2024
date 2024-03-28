import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Chargement des données
df = pd.read_csv('df_final.csv')  # Assurez-vous que le chemin d'accès est correct
X = df.drop(columns=['TARGET'])  # 'TARGET' est supposé être votre colonne cible
y = df['TARGET']

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Initialisation de l'explainer SHAP
explainer = shap.Explainer(model, X_train)

# Calcul des valeurs SHAP pour l'ensemble d'entraînement (peut prendre du temps pour de grands ensembles de données)
shap_values = explainer(X_train)

# Interface Streamlit
st.title("Prédiction de Non-Paiement de Prêt avec XGBoost")

# Explications globales avec SHAP
st.header("Importance Globale des Caractéristiques")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_train, plot_type="bar")
st.pyplot(fig)

# Sélection de l'indice de l'observation à expliquer par un utilisateur
index_to_explain = st.slider("Sélectionnez l'indice de l'observation à expliquer", 0, len(X_test)-1, 0)

# Extraction de l'observation spécifique à expliquer
observation_to_explain = X_test.iloc[index_to_explain:index_to_explain+1]

# Prédiction et probabilité pour l'observation sélectionnée
predicted_class = model.predict(observation_to_explain)[0]
probability_of_default = model.predict_proba(observation_to_explain)[0, 1]  # Probabilité de défaut de paiement

# Affichage des résultats
st.write(f"Prédiction pour l'indice sélectionné {index_to_explain}: {'Non-Paiement' if predicted_class else 'Paiement'}")
st.write(f"Probabilité de non-paiement: {probability_of_default:.4f}")

# Décision basée sur un seuil spécifique
decision_threshold = 0.5  # Ajustez ce seuil selon vos critères
loan_decision = "Prêt Accordé" if probability_of_default < decision_threshold else "Prêt Refusé"
st.write(f"Décision du prêt basée sur le seuil de probabilité de {decision_threshold}: {loan_decision}")

# Affichage des explications SHAP pour l'observation sélectionnée
st.header(f"Explications SHAP pour l'indice sélectionné {index_to_explain}")
shap_values_observation = explainer(observation_to_explain)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values_observation[0], max_display=10)
st.pyplot(fig)
