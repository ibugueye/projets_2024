 
### Manipulation et Analyse des Données
 
import pandas as pd
import numpy as np
 
### Visualisation des Données
 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sweetviz as sv
 

### Machine Learning et Prétraitement
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score,make_scorer, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
 

### Modèles d'Apprentissage Automatique Avancés
 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 

# plotly
 
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
import plotly.figure_factory as ff
 
 

### Rééchantillonnage et Pipelines Avancés
 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
 
### Suivi d'Expériences et Déploiement
 
import mlflow
import streamlit as st
 
from credit_card_default_utils import *


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score, make_scorer
import pandas as pd 
import codecs
 
# Components Pkgs

import streamlit.components.v1 as components
 

import streamlit as st
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import pickle
 
 

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données","EDA automatique" ,
         "MLflow experience"," Modeles Comparés","Modélisation","Choix Classifiers", "Predictions", "shap_explaination"]




page = st.sidebar.radio("Aller vers la page :", pages)
df_train = pd.read_csv("data/application_train.csv")
df_test = pd.read_csv("data/application_test.csv")
df = pd.read_csv("df_final.csv")
df["CODE_GENDER"].fillna(0, inplace=True)
if page == pages[0]:
    st.write('### Contexte du projet')
    st.write("L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.")
    st.image("credit_score.png")

elif page==pages[1]:
    st.write("### Exploration des données ")
    st.dataframe(df.head())
    st.write("Dimensions du dataFrame : ")
    st.write("Nombre de lignes : " ,  df.shape[0])
    st.write("Nombre de colonnes : ",df.shape[1])
    st.write("Missing values")
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
        st.write("Nombre de missing value : ",df.isna().sum().sum())
    if st.checkbox("Afficher les doublons"):
        st.write("Nombre de doublons ",df.duplicated().sum())
       
        
        
elif page == pages[2]:
  
    
    
    st.title(" Analyse de Données")
    
    # Définition de la fonction pour créer un DataFrame des types de données
    
    st.write('## Analyse du Type de Données dans DataFrame')
    def create_dtypes_df(df):
        dtypes_df = pd.DataFrame(df_train.dtypes, columns=['dtype']).reset_index()
        dtypes_df.columns = ['column', 'dtype']
        return dtypes_df
    # Création du graphique en barres
    fig, ax = plt.subplots()
    # Création du DataFrame des types de données
    
    dtypes_df = create_dtypes_df(df_train)
    
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
        data = df_train[column].value_counts(normalize=True)
        data.plot(kind='bar', ax=ax)
        ax.set_title(column)
        
        
    def main():
        st.write('## Analyse Visuelle des Variables Catégorielles')
     
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
 
    st.write()
    
    trace0 = go.Bar(
                x= df[df["TARGET"]== 0]["TARGET"].value_counts().index.values,
                y= df[df["TARGET"]== 0]["TARGET"].value_counts().values,
                name = "Good credit")

    trace1 = go.Bar(
                x= df[df["TARGET"]== 1]["TARGET"].value_counts().index.values,
                y= df[df["TARGET"]== 1]["TARGET"].value_counts().values,
                name = "Bad credit")


    data = [trace0, trace1]
    layout = go.Layout(
            yaxis=dict(
                    title="Count"
            ),

            xaxis= dict(
                title="Risk Variable"
            ),
            title = "Target variable distribution"
    )


    fig = go.Figure(data=data, layout=layout)

    fig.data[0].marker.line.width =4
    fig.data[0].marker.line.color ="black"

    fig.data[1].marker.line.width =4
    fig.data[1].marker.line.color ="black"

    #go.iplot(fig, filename="grouped-bar")


    st.plotly_chart(fig)
      
                
    st.write('## Analyse des distributions ')
# Fonction adaptée pour Streamlit
    def plot_distributions_st(df):
        # Sélectionner uniquement les colonnes numériques
        numeric_columns = df.select_dtypes(include='number').drop(columns=['id'], errors='ignore')  # Ajouté errors='ignore' au cas où 'id' n'existe pas

    # Afficher un histogramme pour chaque colonne numérique
        num_cols = len(numeric_columns.columns)
        rows = (num_cols // 3) + (1 if num_cols % 3 else 0)  # Calculer le nombre de lignes nécessaires

        plt.figure(figsize=(15, 5*rows))  # Ajuster la hauteur en fonction du nombre de lignes

        for i, column in enumerate(numeric_columns.columns):
            plt.subplot(rows, 3, i + 1)
            plt.hist(df[column], bins=20, color='blue', alpha=0.7)
            plt.title(f'Histogramme de {column}')
            plt.xlabel(column)
            plt.ylabel('Fréquence')

        st.pyplot(plt)
        plt.close()
        
        st.write ("Afficher des boîtes à moustaches ")
        plt.figure(figsize=(15, 5*rows))  # Réutiliser le calcul de 'rows' pour la hauteur

        for i, column in enumerate(numeric_columns.columns):
            plt.subplot(rows, 3, i + 1)
            sns.set(style="whitegrid")
            sns.boxplot(x=df[column], palette="Set2")
            plt.title(f'Boîte à moustaches de {column}')
            plt.xlabel(column)

        st.pyplot(plt)
        plt.close()
        
    if __name__ == "__main__":
        plot_distributions_st(df)
        
 
    
elif page==pages[3]: 
    
    st.write("## Analyse exploration automatique avec Sweetviz")   
    def st_display_sweetviz(report_html,width=1000,height=500):
        report_file = codecs.open(report_html,'r')
        page = report_file.read()
        components.html(page,width=width,height=height,scrolling=True)
  
    
    st.write("## Analyse exploration automatique avec Sweetviz")

    # Générer le rapport Sweetviz (cela peut prendre un peu de temps en fonction de la taille des données)
    if st.button('Générer le rapport Sweetviz'):
        report = sv.analyze(df)
        report_path = "sweetviz_report.html"
        report.show_html(report_path)

        # Afficher le rapport dans Streamlit
        st_display_sweetviz(report_path)

 
   

    
        

elif page==pages[4]:
    st.title("Résultats MLflow dans Streamlit")
      
     
    url = "http://127.0.0.1:5000/#/experiments/0/runs/6d1adf1e185f4f3ea4bf9ac1fd91ad26/artifacts"
    st.markdown(f'Accédez à l\'artefact MLflow directement [ici]({url}).', unsafe_allow_html=True)

elif page==pages[5]:
            
    from sklearn.model_selection import train_test_split
    features = df[df.columns.difference(['TARGET'])]
    labels = df["TARGET"]

    # Splitting data into train and test
    X = df.drop(["TARGET"], axis=1)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify =y, random_state=0, train_size = 0.10)

    # Création du DataFrame à partir des résultats fournis
    results_df = pd.DataFrame({
        'Classifier': ['LightGBM', 'XGBoost', 'RandomForest'],
        'Mean ROC AUC': [0.724105, 0.697792, 0.702311],
        'Mean Fit Time': [1.666131, 1.748368, 203.830650],
        'Mean Score Time': [0.024999, 0.035726, 1.857772]
    })
            # Titre de l'application Streamlit
    st.title('Comparaison des Classificateurs')

    # Affichage du DataFrame des résultats
    st.write("## Résultats des Modèles", results_df)

    # Création et affichage d'un graphique pour le Mean ROC AUC
    fig_auc = px.bar(results_df, x='Classifier', y='Mean ROC AUC', text='Mean ROC AUC', color='Classifier',
                    title="Comparaison du Mean ROC AUC des Classificateurs")
    fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_auc)

    # Création et affichage d'un graphique pour le Mean Fit Time
    fig_fit_time = px.bar(results_df, x='Classifier', y='Mean Fit Time', text='Mean Fit Time', color='Classifier',
                        title="Comparaison du Mean Fit Time des Classificateurs")
    fig_fit_time.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_fit_time)

    # Création et affichage d'un graphique pour le Mean Score Time
    fig_score_time = px.bar(results_df, x='Classifier', y='Mean Score Time', text='Mean Score Time', color='Classifier',
                            title="Comparaison du Mean Score Time des Classificateurs")
    fig_score_time.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_score_time) 
    
elif page==pages[6]:
 
    
     
    from joblib import load
    st.title("Modelisation")
    with st.container():
 

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            features = df[df.columns.difference(['TARGET', "SK_ID_CURR"])]
            labels = df['TARGET']
            X = features
            y = labels

            # Créer et entraîner le modèle
            model = RandomForestClassifier()
            model.fit(X, y)

            # Sélection des caractéristiques importantes
            selector = SelectFromModel(model, threshold=0.02)
            features_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            
            
            st.write("features")
            st.write(selected_features)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = RandomForestClassifier(n_estimators=20, max_depth=5)
            clf.fit(X_train, y_train)

            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)

            accuracy_train = accuracy_score(y_train, pred_train)
            accuracy_test = accuracy_score(y_test, pred_test)

            fpr_train, tpr_train, _ = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
            auc_train = auc(fpr_train, tpr_train)

            fpr_test, tpr_test, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
            auc_test = auc(fpr_test, tpr_test)
            
    

           

        with col2:
            st.write("Métriques de formation:")
            st.write("Accuracy:", accuracy_train)
            st.write("AUC:", auc_train)
             

        with col3:
            st.write("Métriques de test:")
            st.write("Accuracy:", accuracy_test)
            st.write("AUC:", auc_test)
             
            
      
        
elif page==pages[7]:  
    # Supposons que `df` a déjà été chargé ici
    # df = pd.read_csv('chemin/vers/votre/fichier.csv')

    # Préparation des données
    X = df.drop(["TARGET"], axis=1)
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.90, random_state=0)

    # Initialisation du modèle
    model = None

    # Sélection et configuration du modèle via la barre latérale Streamlit
    classifier = st.sidebar.selectbox(
        "Choisissez le modèle de classification :",
        ('Random Forest', 'LightGBM', 'XGBoost', 'Régression Logistique', 'Arbre de décision')
    )
    # Afficher le nom du modèle sélectionné comme titre
    st.title(f"Modèle sélectionné : {classifier}")

    # Configuration basée sur le classificateur sélectionné
    if classifier == 'Random Forest':
        n_estimators = st.sidebar.slider("Nombre d'estimateurs", 10, 100, step=10, value=100)
        max_depth = st.sidebar.slider("Profondeur maximale", 1, 20, step=1, value=5)
        bootstrap = st.sidebar.radio("Bootstrap samples ?", ("True", "False"))
        bootstrap = bootstrap == "True"
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        
        
    if classifier == 'LightGBM':
        num_leaves = st.sidebar.slider("num_leaves",31)
        max_depth = st.sidebar.slider("Profondeur maximale", 1, 20, step=1, value=5)
        learning_rate= st.sidebar.slider(0.1,0.9)
        n_estimators = st.sidebar.slider("Nombre d'estimateurs", 10, 100, step=10, value=100)
       
        model = LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth,learning_rate=learning_rate,n_estimators=n_estimators,objective='binary')
  
    if classifier== "XGBoost":
        model = XGBClassifier(
        n_estimators=100,        # Nombre d'arbres boostés à construire
        learning_rate=0.1,       # Étape de réduction pour la mise à jour des poids
        max_depth=3,             # Profondeur maximale de chaque arbre; augmentez pour plus de complexité/modélisation fine
        subsample=1.0,           # Fraction des échantillons à utiliser pour entraîner chaque arbre
        colsample_bytree=1.0,    # Fraction des caractéristiques à utiliser pour chaque arbre
        objective='binary:logistic', # Utilisez 'multi:softprob' pour la classification multiclasse et obtenez des probabilités
        eval_metric='logloss',   # Métrique d'évaluation pour la classification binaire; utilisez 'mlogloss' pour la classification multiclasse
    )
        
    if classifier== "Régression Logistique":
        model = LogisticRegression(
        penalty='l2',    # Régularisation L2 par défaut
        C=1.0,           # Inverse de la force de régularisation; des valeurs plus faibles spécifient une régularisation plus forte
        solver='lbfgs',  # Bon choix pour de petits ensembles de données et pour la classification multiclasse
        max_iter=100,    # Augmentez-le si l'optimisation n'a pas convergé (warnings)
        multi_class='auto' # Choix automatique entre 'ovr' (One-vs-Rest) et 'multinomial'
    )
            
      
        # Configurez et entraînez les autres modèles de manière similaire ici

    if model is not None:
        # Entraînement du modèle
        model.fit(X_train, y_train)
        # Prédiction
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calcul et affichage des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        
        
        selected_metrics = st.multiselect(
        "Sélectionnez les métriques à visualiser :",
        options=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
        default=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"]
)

       
                
        # Supposons que les variables 'accuracy', 'precision', 'recall', 'f1', et 'roc_auc' sont déjà définies
        # avec les valeurs des métriques de performance du modèle

        # Création d'un DataFrame avec les métriques
        metrics_data = {
            "Métrique": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
            "Valeur": [accuracy, precision, recall, f1, roc_auc]
        }
        metrics_df = pd.DataFrame(metrics_data)

        # Filtrer le DataFrame pour inclure uniquement les métriques sélectionnées
        filtered_metrics_df = metrics_df[metrics_df["Métrique"].isin(selected_metrics)]

        # Affichage du DataFrame filtré
        st.table(filtered_metrics_df)


        # https://medium.com/towardsdev/machine-learning-algorithms-6-metrics-for-binary-classification-faf0db9b5ad8
        
        
                
elif page==pages[8]:  
    
            
    

    # URL of the Flask API
    API_URL = 'http://127.0.0.1:5000/predict'  # Adjust this to the URL of your Flask API

    # Interface utilisateur de l'application
    st.title("Application de Prédiction de Défaut de Paiement")
    
    

    # Collecting input data through sliders in the sidebar
    
    st.sidebar.header("Informations sur le client")
     
    input_data = {
        'SK_ID_CURR': st.sidebar.slider('SK_ID_CURR', min_value=100000, max_value=999999, value=100001, step=1),
        'CNT_CHILDREN': st.sidebar.slider('CNT_CHILDREN', min_value=0, max_value=20, value=0, step=1),
        'AMT_INCOME_TOTAL': st.sidebar.slider('AMT_INCOME_TOTAL', min_value=20000, max_value=1000000, value=100000, step=1000),
        'AMT_CREDIT': st.sidebar.slider('AMT_CREDIT', min_value=50000, max_value=2000000, value=500000, step=10000),
        # Add other features as necessary...
    }

    # Button to make prediction
    if st.button('Prédire le risque de défaut'):
        # Sending the data to the Flask API and getting the prediction
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            # Extracting the prediction results
            result = response.json()
            probabilities = result['probabilities']
            prediction = result['prediction'][0]  # Assuming the first (and only) prediction

            # Assuming `probabilities` is the probability of default returned by your API
            probability_of_default = probabilities[0]  # The API returns a list with a single probability value
            probability_of_no_default = 1 - probability_of_default  # Compute the probability of no default

            # Now `probabilities` should contain both values, matching the length of `categories`
            probabilities = [probability_of_no_default, probability_of_default]

            # Displaying the results in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Résultat de la prédiction :")
                result_message = "Le client sera en défaut de paiement." if prediction == 1 else "Le client ne sera pas en défaut de paiement."
                st.write(result_message)

                st.subheader("Probabilités :")
                # Plotting the probability as a bar chart
                categories = ['Pas de défaut', 'Défaut']
                fig = px.bar(x=categories, y=probabilities, labels={'x': '', 'y': 'Probabilité'}, title="Probabilité de Défaut de Paiement")
                st.plotly_chart(fig)
      

elif page==pages[9]: 
    
    st.title("L’interprétabilité globale et locale du modèle ") 
    
    
    df = pd.read_csv("df_final.csv")

                                            
                # Splitting data into train and test
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify =y, random_state=0, train_size = 0.10)
                    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, train_size=0.10)

                    #s Entraînement d'un modèle XGBoost (ou tout autre modèle)
    model = xgb.XGBClassifier().fit(X_train, y_train)

                # Création de l'expliqueur SHAP et calcul des valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
                
    
    # SHAP Summary Plot pour une interprétation globale
    st.write("### Interpretation Globale")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(plt, bbox_inches='tight', pad_inches=0)
    plt.clf()  # Efface la figure courante pour réutiliser plt pour un autre graphique
          
    st.write("### Interpretation locale")
                    # SHAP Waterfall Plot pour une interprétation locale d'une instance spécifique
                    # Note : pour des raisons de démonstration, je choisis l'indice 0, mais vous pouvez le modifier ou le rendre dynamique
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=True)
    st.pyplot(plt, bbox_inches='tight', pad_inches=0)
    plt.clf()