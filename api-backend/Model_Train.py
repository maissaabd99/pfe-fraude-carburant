# ----------------------------------- Isolation Forest ---------------------------
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Data_Preprocessing import Data_Preprocessing
class Model_Train:

    def __init__ (self):
        print("shy")
        self.minmaxscaler = MinMaxScaler()

    #get training data
    def import_data(self,pathdata):
        dataframe = pd.read_csv(pathdata, delimiter=",")
        dataframe["date"] = pd.to_datetime(dataframe["date"], format="%d/%m/%Y")
        print(dataframe.head())
        return dataframe

    #train isolation forest models and save them
    def train_isolation_forest_model(self, mat_dataset):
        string_model = "models/" + str(mat_dataset["matricule"].unique()) + ".pkl"
        data = mat_dataset[['quantite','type_carburant_enc','typePaiement_enc','diff_date','gouvernorat_enc']]
        np_scaled = self.minmaxscaler.fit_transform(data)
        # Isolation forest model
        ifo_model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            max_features=4,
            bootstrap=False,
            warm_start=True,
            verbose=1,
            random_state=0)
        ifo_model.fit(np_scaled)
        mat_dataset["anomaly_if"] = ifo_model.predict(data)
        mat_dataset["anomaly_score"] = ifo_model.decision_function(data)
        joblib.dump(ifo_model, string_model)
        return mat_dataset, ifo_model

    def execute_model_training (self) :
        # récuperer la liste des matricules
        dataframe = self.import_data("data/output-data/gp2_carburant_transformed_diff_date_3.csv")
        matricules = dataframe["matricule"].unique()
        if_final = pd.DataFrame()
        print(if_final)
        m = 0
        for mat in matricules:
            data_per_mat = dataframe[dataframe["matricule"] == mat]
            if_dataset, m = IsolationForest.train_isolation_forest_model(data_per_mat)
            if_final = pd.concat([if_final, if_dataset], ignore_index=True)
        # enregistrer le dataset annoté
        if_final.to_csv("data/output-data/if_dataset.csv", index=False)
        
#-------------------------------Evaluation des performances de Isolation Forest----------------------
    def if_evaluation (self) : 
        my_saved_model = joblib.load("models/[230].pkl")
        transactions = pd.DataFrame(
            {
                'matricule': [230, 230, 230,230,230, 230, 230, 230, 230,230,230,230,230
                ,230,230,230,230,230,230,230],
                "quantite": [
                    52.09,56.65,56.65,68.09,68.09,444,50.0,75.0,30,
                    43.00,76.0,80.33,82.98,100.00,98.08,25.09,30,49.00,70.9,75.00,
                ],
                "type_carburant": [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
                "typePaiement": [
                    31,32,32,32,32,32,32,
                    32,32, 32,30,32,32,32,32,32,32,32,32,32,
                ],
                'fournisseur_enc' : [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "diff_date": [2.0,2.0,2.0,2.0,1.0,2.0,1.0,2.0,.0,2.0,2.0,2.0,2.0,1.0,2.0, 0.0,2.0,4.0,2.0,3.0,
                ], 
            }
        )
        transactions.to_csv("data/validation_set_230.csv", index=False)
        np_scaled = self.minmaxscaler.fit_transform(transactions)
        # np_scaled = pd.DataFrame(np_scaled)
        transactions["anomaly_if"] = my_saved_model.predict(np_scaled)
        transactions["real_labels"] = [-1,-1,1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1]
        print(transactions)
            
        # Prédictions de votre modèle
        y_pred = transactions["anomaly_if"]
        # Calcul de l'accuracy
        y_true = transactions["real_labels"]
        accuracy = accuracy_score(y_true, y_pred)
        # Calcul de la précision
        precision = precision_score(y_true, y_pred)
        # Calcul du rappel
        recall = recall_score(y_true, y_pred)
        # Calcul du F1 Score
        f1 = f1_score(y_true, y_pred)
        
        print("Précision :", precision)
        print("\n")
        print("Recall :", recall)
        print("\n")
        print("Accuracy :", accuracy)
        print("F1 Score :", f1)
    
    #get inference of a transaction
    def inference(self, observation) : 
        print("observation recu comme dataframe :",observation)
        alldata = Data_Preprocessing.getTransformedData()
        observation = observation.iloc[0]
        print("observation avec iloc :",observation)
        matricule = observation['matricule']

        df_mat = alldata[alldata['matricule'] == numpy.int64(matricule)]
        #get dernire transaction
        df_mat['date'] = pd.to_datetime(df_mat['date'], format='%Y/%m/%d')
        df_mat['date'] = df_mat['date'].sort_values()
        last_obs = df_mat.iloc[-1]
        last_obs = last_obs[["matricule","quantite", "type_carburant", "typePaiement", "date","gouvernorat","fournisseur"]]
        
        #concatener les 2 transactions pour calculer diff_date
        new_data_frame = pd.DataFrame([last_obs,observation])
        new_data_frame['date'] = pd.to_datetime(new_data_frame['date'], format='%Y/%m/%d')
        #new_data_frame['date'] = new_data_frame['date'].sort_values()
        print("New data frame after sort :",new_data_frame)
        print("index of last obs = ",last_obs.name)
        difference = new_data_frame.loc[0, 'date'] - new_data_frame.loc[last_obs.name, 'date']
        observation['diff_date'] = difference.days
        
        observation['gouvernorat_enc'] = df_mat.loc[df_mat["gouvernorat"] == observation['gouvernorat'],"gouvernorat_enc"]       
        observation['fournisseur_enc'] =  df_mat.loc[df_mat["fournisseur"] == observation['fournisseur'], "fournisseur_enc"]
        observation['type_carburant_enc'] =  df_mat.loc[df_mat["type_carburant"] == observation['type_carburant'], "type_carburant_enc"]
        observation['typePaiement_enc'] =  df_mat.loc[df_mat["typePaiement"] == observation['typePaiement'], "typePaiement_enc"]
        print('gouvernorat matching  est : \n',observation)

        if len(observation['fournisseur_enc']) > 0:
            observation['fournisseur_enc'] = observation['fournisseur_enc'].iloc[0]
        else:
            plus_grande_valeur = df_mat["fournisseur_enc"].max()
            valeur_par_defaut = plus_grande_valeur + 1
            observation['fournisseur_enc'] = valeur_par_defaut

        if len(observation['gouvernorat_enc']) > 0 :
            observation['gouvernorat_enc'] = observation['gouvernorat_enc'].iloc[0]
        else:
            plus_grande_valeur = df_mat["gouvernorat_enc"].max()
            valeur_par_defaut = plus_grande_valeur + 1
            observation['gouvernorat_enc'] = valeur_par_defaut
        
        if len(observation['type_carburant_enc']) > 0:
            observation['type_carburant_enc'] = observation['type_carburant_enc'].iloc[0]
        else:
            plus_grande_valeur = df_mat["type_carburant_enc"].max()
            valeur_par_defaut = plus_grande_valeur + 1
            observation['type_carburant_enc'] = valeur_par_defaut
        
        if len(observation['typePaiement_enc']) > 0:
            observation['typePaiement_enc'] = observation['typePaiement_enc'].iloc[0]
        else:
            plus_grande_valeur = df_mat["typePaiement_enc"].max()
            valeur_par_defaut = plus_grande_valeur + 1
            observation['typePaiement_enc'] = valeur_par_defaut
        
        observation = observation[['quantite','type_carburant_enc','diff_date','gouvernorat_enc',
                        'fournisseur_enc']]
        my_model = joblib.load("models/["+str(matricule)+"].pkl")
        print('my loader model is :', my_model)
        observation = pd.DataFrame([observation])
        print("observation after encoding :")
        print(observation)
        np_scaled = self.minmaxscaler.fit_transform(observation)
        # np_scaled = pd.DataFrame(np_scaled)
        observation["anomaly_if"] = my_model.predict(observation)
        observation["matricule"] = numpy.int64(matricule)
        print("observation after anoamly detection :")
        print(observation)
        return observation