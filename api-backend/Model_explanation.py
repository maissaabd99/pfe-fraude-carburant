#----------------------------- Explainable AI for Isolation Forest -----------------------

from lime.lime_tabular import LimeTabularExplainer
import joblib
import pandas as pd
import Data_Preprocessing

class Model_Explanation() :
  
  def explain_observation(observation) :
        print("Observation recu est : ",observation)
        #observation = pd.DataFrame([observation])
        matricule = observation['matricule'].iloc[0]
        print('matricule received by inference :',matricule)
        observation = observation[['quantite','type_carburant_enc','diff_date','gouvernorat_enc',
                        'fournisseur_enc']]
        my_saved_model = joblib.load("models/["+str(matricule)+"].pkl")
        #my_saved_model = joblib.load("model_global.pkl")
        dataframe = Data_Preprocessing.Data_Preprocessing.getTransformedData()
        sub_mat = dataframe[dataframe['matricule'] == matricule]
        sub_mat = sub_mat[['quantite','type_carburant_enc','diff_date','gouvernorat_enc',
                        'fournisseur_enc']]
        test_sub_mat_for_lime = sub_mat
        #print(sub_mat)    
        #y_train = sub_mat['anomaly_if']
        #test_sub_mat_for_lime = test_sub_mat_for_lime.drop(['anomaly_if','matricule','anomaly_score'],axis=1)
        # Obtenir les scores d'anomalie (plus le score est bas, plus la probabilité d'être une anomalie est élevée)
        scores = my_saved_model.decision_function(test_sub_mat_for_lime)
        feature_names = test_sub_mat_for_lime.columns.tolist()
        # Créez un explainer LIME
        explainer = LimeTabularExplainer(test_sub_mat_for_lime.values, 
                                         mode="regression", 
                                         training_labels=scores, 
                                         feature_names=feature_names,
                                         class_names=[-1, 1])
        # Choisissez un échantillon d'anomalie
        #sample_index = 6
        #sample = test_sub_mat_for_lime.iloc[103]
        # Obtenez l'explication LIME
        explanation = explainer.explain_instance(observation.iloc[0], my_saved_model.decision_function,num_features=5)
        important_features = explanation.as_list()
        # Afficher les caractéristiques importantes
        print("Caractéristiques importantes pour la prédiction de l'anomalie : \n \n")
        for feature in important_features:
            feature_name = feature[0]  # Nom de la caractéristique
            feature_weight = feature[1]  # Poids de la caractéristique dans la prédiction
            print(f"- {feature_name}: {feature_weight}")
        # Le feature qui a affecté le plus le résultat de sub modèle de IF
        most_contributed_feature = min(important_features, key=lambda x: x[1])
        #most_contributed_feature = min(important_features[1])
        return important_features
