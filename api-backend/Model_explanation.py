#----------------------------- Explainable AI for Isolation Forest -----------------------

import joblib
import pandas as pd
import Data_Preprocessing
import shap

class Model_Explanation() :

  def explain_observation(observation):
    print("Observation reçue est : ", observation)
    matricule = observation['matricule'].iloc[0]
    print('Matricule reçu par inference :', matricule)
    
    observation = observation[['quantite', 'type_carburant_enc', 'diff_date', 'gouvernorat_enc']]
    my_saved_model = joblib.load("models/[" + str(matricule) + "].pkl")
    print("ghhhhhhhhhhhhhhhhhhh",my_saved_model)
    
    dataframe = Data_Preprocessing.Data_Preprocessing.getTransformedData()
    sub_mat = dataframe[dataframe['matricule'] == matricule]
    sub_mat = sub_mat[['quantite', 'type_carburant_enc', 'diff_date', 'gouvernorat_enc']]
    
    explainer = shap.TreeExplainer(my_saved_model)
    shap_values = explainer.shap_values(observation)
    
    # Associer les noms des caractéristiques avec les valeurs SHAP
    important_features = list(zip(observation.columns, shap_values[0]))
    
    # Afficher les caractéristiques importantes
    print("Caractéristiques importantes pour la prédiction de l'anomalie : \n")
    for feature, weight in important_features:
        print(f"- {feature}: {weight}")
    
    return important_features
  

