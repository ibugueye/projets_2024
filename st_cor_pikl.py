import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import joblib  # Pour charger le modèle

# Chargement du modèle sérialisé
model_path = 'best_model.joblib'
pipeline = joblib.load(model_path)

# Extraire le modèle de classification du pipeline
# Assurez-vous que le nom de l'étape dans votre pipeline correspond à ce que vous avez défini lors de sa création
classifier_model = pipeline.named_steps['classifier']

# Préparation des données (comme précédemment)
df = pd.read_csv('df_final.csv')  # Remplacez par votre chemin de fichier correct
X = df.drop(columns=['TARGET'])
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialiser l'explainer SHAP avec le modèle extrait
explainer = shap.Explainer(classifier_model, X_train)
# Calcul des valeurs SHAP pour l'ensemble d'entraînement (pour les explications globales)
shap_values = explainer(X_train)

# Interface Streamlit
st.title("Prédiction de Non-Paiement de Prêt avec le Meilleur Modèle")

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
predicted_class =classifier_model .predict(observation_to_explain)[0]
probability_of_default = classifier_model.predict_proba(observation_to_explain)[0, 1]  # Probabilité de défaut de paiement

# Affichage des résultats
st.write(f"Prédiction pour l'indice sélectionné {index_to_explain}: {'Non-Paiement' if predicted_class else 'Paiement'}")
st.write(f"Probabilité de non-paiement: {probability_of_default:.4f}")

# Décision basée sur un seuil spécifique
decision_threshold = 0.428  # Ajustez ce seuil selon vos critères
loan_decision = "Prêt Accordé" if probability_of_default < decision_threshold else "Prêt Refusé"
st.write(f"Décision du prêt basée sur le seuil de probabilité de {decision_threshold}: {loan_decision}")

# Affichage des explications SHAP pour l'observation sélectionnée
st.header(f"Explications SHAP pour l'indice sélectionné {index_to_explain}")
shap_values_observation = explainer(observation_to_explain)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values_observation[0], max_display=10)
st.pyplot(fig)
