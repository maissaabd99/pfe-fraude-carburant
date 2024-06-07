import pandas as pd
import matplotlib
import re
#import Shell_Web_Scrapping as ws
#from getGovFromZone import get_gouvernorat
from random import choice
from sklearn.preprocessing import LabelEncoder
import pickle

#---------------------------------------**Données des bus CAN**------------------------------
matplotlib.use('Agg')

class Data_Preprocessing ():
    def __init__ (self):
        print("shy")

    #self : pour inquer pour indiquer qu'elle appartient à l'instance actuelle de la classe.
    def drop_columns(self, df, columns_names) :
        df.drop(columns=[col for col in columns_names if col in df.columns], axis=1, inplace=True)

    def process_data(self, filepath) :
        gp2_carburant = pd.read_csv(filepath, delimiter=",")
        print(gp2_carburant.sample(5))
        print(gp2_carburant.shape) 
        # ### Vérification des données manquantes
        print(gp2_carburant.isnull().sum())
        
        #--------------- I - Analyse et prétraitement des caractéristiques---------------
        cols = [
            "is_modified_qty","is_modified_date","is_modified_odometre",
            "is_modified_coutL","lon","lat","numPaiement",
        ]
        #gp2_carburant = gp2_carburant.drop(["is_modified_qty","is_modified_date","is_modified_odometre","is_modified_coutL","lon","lat",
        #        "numPaiement",],axis=1)
        self.drop_columns(gp2_carburant, cols)
        
        # Vérifions la suppresion
        print(gp2_carburant.head())
                
        #La valeur 0 peut être due à un problème d'extraction des données provenant des bus CAN, donc nous allons la rejeter pour le moment. </li>
        #gp2_carburant = gp2_carburant.drop(["odometre"], axis=1)
        self.drop_columns(gp2_carburant,["odometre"])
        
        #--------------------------- 1 - Caractéristiques : CoutT et CoutL et id-------------------
        
        #gp2_carburant = gp2_carburant.drop(["coutT", "coutL", "id"], axis=1)
        self.drop_columns(gp2_carburant,["coutT", "coutL", "id"])
        print(gp2_carburant.head())
       
        # renommer la colonne note en type_carburant
        gp2_carburant = gp2_carburant.rename(columns={"note": "type_carburant"})
        
        # * Changement à faire : remplacer 'Gasoil' par 'Gazoil'
        sub_gasoil = gp2_carburant[gp2_carburant["type_carburant"] == "Gasoil"]
        print(sub_gasoil)
                
        # fusionner
        gp2_carburant["type_carburant"].replace("Gasoil", "Gazoil", inplace=True)     
                 
        #--------------------- 2 - Caractéristique : Station ------------------------------
        
        #data_with_zone = ws.Web_scrapping_Station.web_scrapping_station(gp2_carburant)
        #data_with_gouvernorat = get_gouvernorat(data_with_zone)
        #gp2_carburant = data_with_gouvernorat
       
        #------------------------------ 3 - Caractéristique : Matricule -----------------------------------
        nbr = gp2_carburant["matricule"].unique()
        print("Nous avons ", len(nbr), "véhicule distincts")
        
        # nombre d'observation par véhicule
        print(gp2_carburant["matricule"].value_counts())
             
        # Regroupement des matricules inconnues
        data = gp2_carburant        
        matricules = data["matricule"].unique()
        df_matricule = data[data["matricule"] == 111111111]
        df_matricule
        new_data = pd.DataFrame()
        for numcarte in df_matricule["typePaiement"].unique():
            data_aux = df_matricule[df_matricule["typePaiement"] == numcarte]
            newmat = choice([i for i in range(100, 1000) if i not in matricules])
            data_aux["matricule"] = newmat
            new_data = pd.concat([new_data, data_aux], ignore_index=True)
        
        new_data
        
        # maintenant remplacer la partie transformée dans le dataset original
        data = data[data["matricule"] != 111111111]
        data = pd.concat([data, new_data], ignore_index=True)
        
        # avant les transformation
        print("Nombre de véhicules avant la transformation : ",len(gp2_carburant["matricule"].unique()))
        # après la transformation
        print("Nombre de véhicules après la transformation : ", len(data["matricule"].unique()))
        
        gp2_carburant = data 
        print(gp2_carburant)
        
        #------ 4 - Trouver la différence en jours entre les date consécutives - diff_date--------        
        gp2 = gp2_carburant
        # convertir en datetime la varibale date
        gp2["date"] = pd.to_datetime(gp2["date"], format="%Y/%m/%d %H:%M").dt.date
        print(gp2["date"])
        
        # Trier les données par matricule puis par date (au cas où ce n'est pas déjà fait)
        gp2 = gp2.sort_values(by=["matricule", "date"]) 
        print(gp2.head())
        
        # Trouver la différence la plus fréquente
        all_matricules = gp2["matricule"]
        all_matricules = all_matricules.unique()
        new_dataframe = pd.DataFrame()
        
        for mat in all_matricules:
            sub_data_frame = gp2[gp2["matricule"] == mat]
            dates = sub_data_frame["date"]
            dates = dates.sort_values()
            # Calculer les différences entre les dates consécutives en jours
            differences = dates.diff().dt.days.dropna()
            # Trouver la différence la plus fréquente
            most_frequent_period = differences.mode()[0]
            print(most_frequent_period)
            sub_data_frame = sub_data_frame.assign(diff_date = differences)
            # recuperer la première ligne
            first_obs = sub_data_frame[0:1]
            print("before : ", sub_data_frame["diff_date"][0:1])
            sub_data_frame["diff_date"][0:1] = most_frequent_period
            print("After : ", sub_data_frame["diff_date"][0:1])
            print(sub_data_frame[0:1])
        
            new_dataframe = pd.concat([new_dataframe, sub_data_frame], ignore_index=True) 
        gp2_carburant = new_dataframe
        print(gp2_carburant.isnull().sum())
        
        #--------------------------------- 5 - Caratéristique : typePaiment-------------------------------
        gp2_carburant["typePaiement"]
        # * Nous allons extraire juste le numéro de la carte (valeur numérique)
        def extraire_nombres(chaine):
            nombres = re.findall(
                r"\d+", chaine
            )  # Utilisation d'une expression régulière pour extraire les nombres
            return [int(nombre) for nombre in nombres]  # Convertir les nombres en entiers
        
        gp2_carburant["typePaiement"] = gp2_carburant["typePaiement"].apply(
            lambda x: extraire_nombres(x))
        gp2_carburant["typePaiement"] = gp2_carburant["typePaiement"].apply(
            lambda x: x[0] if len(x) > 0 else None)     
        print(gp2_carburant.head())
        
        #--------------------------- II - Encodage des variables catégorielles----------------- 
        gp2_carburant = self.data_encoding(gp2_carburant)        
        gp2_carburant.to_csv("gp2_carburant_transformed_diff_date_3.csv", index=False)
        return gp2_carburant
    
    #encodage des variables catégorielles
    def data_encoding(gp2_carburant) :
        label_encoders = {}
        le_gouvernorat = LabelEncoder()
        le_carburant = LabelEncoder()
        le_fournisseur = LabelEncoder()
        #gp2_carburant['station_enc'] = le_station.fit_transform(gp2_carburant['station'])
        gp2_carburant['type_carburant_enc'] = le_carburant.fit_transform(gp2_carburant['type_carburant'])
        gp2_carburant['fournisseur_enc'] = le_fournisseur.fit_transform(gp2_carburant['fournisseur'])
        gp2_carburant['gouvernorat_enc'] = le_gouvernorat.fit_transform(gp2_carburant['gouvernorat'])

        label_encoders["type_carburant"] = le_carburant
        label_encoders["gouvernorat"] = le_gouvernorat
        label_encoders["fournisseur"] = le_fournisseur

        # Enregistrement des encodeurs dans un fichier
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        return gp2_carburant
   
   #get annotated dataset with isolation forest 
    def getTransformedData ():
        data =pd.read_csv("data/output-data/if_dataset.csv")
        return data
    
#if __name__ == '__main__': 
#    data = pd.read_csv("data/output-data/gp2_carburant_transformed_diff_date_2.csv", delimiter=",")
#df = Data_Preprocessing.data_encoding()
#    print(df.head())