import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Charger les données
# Assurez-vous que le chemin d'accès est correct et que les données sont prêtes pour la modélisation
df = pd.read_csv('df_final.csv')  # Mettez ici le chemin correct vers votre fichier de données
X = df.drop(columns=['TARGET'])  # Supposons que 'TARGET' est votre variable cible
y = df['TARGET']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Formation du modèle (cela suppose que vous avez déjà choisi un modèle après votre exploration)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Interface Streamlit
st.title("Prédiction de Non-Paiement de Prêt avec XGBoost")

# Sélection de l'indice de l'observation à expliquer par un utilisateur
index_to_explain = st.slider("Sélectionnez l'indice de l'observation à expliquer", 0, len(X_test)-1, 0)

# Extraction de l'observation spécifique à expliquer
observation_to_explain = X_test.iloc[index_to_explain:index_to_explain+1]

# Prédiction et probabilité pour l'observation sélectionnée
predicted_class = model.predict(observation_to_explain)[0]
probability_of_default = model.predict_proba(observation_to_explain)[0, 1]  # Probabilité de défaut de paiement

# Afficher les résultats
st.write(f"Prédiction pour l'indice sélectionné {index_to_explain}: {'Non-Paiement' if predicted_class else 'Paiement'}")
st.write(f"Probabilité de non-paiement: {probability_of_default:.4f}")

# Décision basée sur un seuil spécifique
decision_threshold = 0.3  # Vous pouvez ajuster ce seuil selon vos critères
loan_decision = "Prêt Accordé" if probability_of_default < decision_threshold else "Prêt Refusé"
st.write(f"Décision du prêt basée sur le seuil de probabilité de {decision_threshold}: {loan_decision}")
