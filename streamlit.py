# Importations des Librairies de Base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings

# Configurations
warnings.filterwarnings('ignore')

# Importations Scikit-learn pour le Prétraitement et la Modélisation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Importations pour les Modèles d'Apprentissage Automatique Avancés
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 

# Importations Imbalanced-learn pour le Rééchantillonnage
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline


import streamlit as st
df_cleaned    = pd.read_csv('df_final.csv')
df_train = pd.read_csv('data/application_train.csv')
df= df_train.copy()

 

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)


if page == pages[0]:
    st.write('### Contexte du projet')
    st.write("L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.")
    #st.image("image.jpg")
    
elif page==pages[1]:
    st.write("### Exploration des données ")
    st.dataframe(df.head())
    st.write("Dimensions du dataFrame : ")
    st.write(df.shape)
    
    # La fonction pour calculer les valeurs manquantes
    def missing_values_table(df_train):
        mis_val = df_train.isnull().sum()
        mis_val_percent = 100 * df_train.isnull().sum() / len(df_train)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        return mis_val_table_ren_columns

    # Titre de l'application
    st.title('Analyse des Valeurs Manquantes')

    # Calcul des valeurs manquantes
    missing_values = missing_values_table(df_train)
    
    # Afficher le tableau des valeurs manquantes
    st.write("Tableau des Valeurs Manquantes :")
    st.table(missing_values)
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
    if st.checkbox("Afficher les doublons"):
        st.write(df.duplicated().sum())
        
elif page == pages[2]:
    st.title("Modélisation avec XGBoost et SHAP")
    
    
    st.title(" Analyse de Données")
    
    # Définition de la fonction pour créer un DataFrame des types de données
    
    st.write('## Analyse du Type de Données dans DataFrame')
    def create_dtypes_df(df):
        dtypes_df = pd.DataFrame(df.dtypes, columns=['dtype']).reset_index()
        dtypes_df.columns = ['column', 'dtype']
        return dtypes_df
    # Création du graphique en barres
    fig, ax = plt.subplots()
    # Création du DataFrame des types de données
    
    dtypes_df = create_dtypes_df(df)
    
    sns.barplot(data=dtypes_df
                .groupby("dtype", as_index=False)["column"]
                .count()
                .sort_values("column", ascending=False)
                .rename({"column": "count"}, axis=1), 
                x="dtype", y="count", ax=ax)

    # Ajout des étiquettes sur les barres
    ax.bar_label(ax.containers[0], fontsize=10)

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    
    
    # Exemple de fonction create_cat_bar_plot (à adapter selon votre mise en œuvre réelle)
    def create_cat_bar_plot(column, ax):
    # Votre logique de dessin de graphique ici, par exemple:
        data = df[column].value_counts(normalize=True)
        data.plot(kind='bar', ax=ax)
        ax.set_title(column)
        
        
    def main():
        st.title('Analyse Visuelle des Variables Catégorielles')
     
        # Création de la figure et des axes pour les sous-graphiques
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
        fig.tight_layout(pad=2)

        # Appel de votre fonction pour chaque sous-graphique
        create_cat_bar_plot("NAME_CONTRACT_TYPE", axes[0, 0])
        create_cat_bar_plot("CODE_GENDER", axes[0, 1])
        create_cat_bar_plot("FLAG_OWN_CAR", axes[1, 0])
        create_cat_bar_plot("FLAG_OWN_REALTY", axes[1, 1])

        # Ajouter une légende pour chaque graphique
        for i in range(4):
            fig.get_axes()[i].legend(title="TARGET", loc="upper right")

        # Afficher la figure dans Streamlit
        st.pyplot(fig)

    if __name__ == "__main__":
        main()
 
elif page == pages[3]:
    
    
    
    
    import streamlit as st
    import pandas as pd
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Assurez-vous que shap est importé correctement
    import shap

    # Charger vos données
    # Assurez-vous que le chemin d'accès à votre fichier est correct
    df = pd.read_csv('df_final.csv')
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Split des données et formation du modèle
    model = XGBClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    # Remplissage des valeurs manquantes pour 'CODE_GENDER' dans X_test si nécessaire
    X_test['CODE_GENDER'] = X_test['CODE_GENDER'].fillna(0)
    model.fit(X_train, y_train)

    # Utiliser SHAP pour expliquer le modèle
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train.iloc[:100, :])
 
    # Streamlit UI
    st.title("Analyse SHAP avec Streamlit")
    
    # Affichage du résumé plot pour les premières 100 observations
    st.header("Résumé Plot pour les premières 100 observations")

    # Sélection de l'indice de l'observation à expliquer
    index_to_explain = st.slider("Sélectionnez l'indice de l'observation à expliquer", 0, 99, 0)

    # Afficher les valeurs SHAP pour une observation spécifique
    st.header(f"Valeurs SHAP pour l'observation {index_to_explain}")
    st.bar_chart(shap_values.values[index_to_explain])

    
    shap.summary_plot(shap_values, X_train.iloc[:100, :])

    # Pour afficher le résumé plot dans Streamlit, vous pouvez utiliser matplotlib
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_train.iloc[:100, :], show=False)
    st.pyplot(fig)
 
 
  

    # Expliquer le modèle avec SHAP
  
    # Afficher les valeurs SHAP pour une observation spécifique
  

    # Prédictions (scores de probabilité de non-paiement)
    st.header("Prédiction de Score de Non-Paiement de Prêt")
    # Calculer les probabilités de prédiction
    probabilities = model.predict_proba(X_test.iloc[index_to_explain:index_to_explain+1, :])
    # Afficher la probabilité de non-paiement (classe 1)
    st.write(f"Probabilité de non-paiement du prêt : {probabilities[0][1]:.2f}")
    
    
    def votre_fonction_de_recherche(sk_id_curr, df):
            # Trouver la ligne correspondant à sk_id_curr
            row = df[df['SK_ID_CURR'] == sk_id_curr]
            
            # Assurez-vous de retirer 'SK_ID_CURR' et la colonne cible si elles sont encore présentes
            features = row.drop(['SK_ID_CURR', 'TARGET'], axis=1)
            
            # Vérifiez si la ligne existe
            if not features.empty:
                # Transformer en array pour le modèle si nécessaire
                return features.values.reshape(1, -1)
            else:
                return None


    # Interface Streamlit
    st.title("Prédiction de Non-Paiement de Prêt")

    # Entrée de l'utilisateur pour SK_ID_CURR
    sk_id_curr = st.number_input("Entrez le SK_ID_CURR pour prédiction:", value=100001, step=1)

    if st.button("Prédire"):
        if sk_id_curr > 0:
            features = votre_fonction_de_recherche(sk_id_curr, df)  # Assurez-vous que df est votre DataFrame complet
            if features is not None and features.shape[1] == 36:
                prediction = model.predict(features)
                probabilities= model.predic_proba(features)
                seuil = 0.65
                decision = 'Non defaut' if probabilities[0][1] < seuil else 'Defaut'
                
                st.write(f"Probabilité de non-paiement: {probabilities[0][1]:.2f}")
                st.write(f"Décision: {decision}")
            else:
                st.write("ID non trouvé ou features incorrectes.")
        else:
            st.write("Veuillez entrer un ID valide.")
        
