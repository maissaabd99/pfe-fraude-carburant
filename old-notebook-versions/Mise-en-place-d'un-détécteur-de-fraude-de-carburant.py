# # Mise en place d'un détecteur de fraude de carburant

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask

gp2_carburant = pd.read_csv('/Users/abdelwahed/Downloads/gp2_carburant (4) (1).csv',delimiter=';', encoding='utf-8')
gp2_carburant.sample(5)
gp2_carburant.shape

# ### Liste de noms de colonnes 

colonnes = gp2_carburant.columns
nbrcolonnes = len(gp2_carburant.columns)
print('Liste des caractéristiques dans le jeu de données \n', colonnes)
print('Nombre de caractéristiques dans le jeu de données : ', nbrcolonnes)

# ### Vérification des données manquantes

print(gp2_carburant.isnull().sum())

# ### Analyse des caractéristiques

cols = ['is_modified_qty','is_modified_date','is_modified_odometre','is_modified_coutL','lon','lat','numPaiement']
for col in cols :
    print("Les valeurs uniques de la colonne : ",col," : ",gp2_carburant[col].unique())

# #### <li> Toutes les valeurs de ces 7 colonnes sont des 0 ou NULL, donc nous allons les suprrimer</li> 

gp2_carburant = gp2_carburant.drop(['is_modified_qty','is_modified_date','is_modified_odometre','is_modified_coutL','lon','lat','numPaiement'],axis = 1)
#Vérifions la suppresion 
gp2_carburant.head()

# ####  <font color=blue>**Caractéristique : Odometre** </font>

import matplotlib.pyplot as plt
import numpy as np

# La colonne odometre
freq_0 = gp2_carburant['odometre'].value_counts()[0]
freq = freq_0 * 100 / 2896
print(freq_0,"enregistrements avec 0 comme valeur dans la colonne Odometre avec un pourcentage de : ",freq,"%")

couleurs = ['lightgreen', 'lightgrey']
plt.pie([100-freq,freq], autopct='%1.0f%%', colors = couleurs,labels = ['valeus existantes', 'O comme valeur'])

plt.title('Fréquence de la valeur 0 dans Odometre', fontsize=12)

plt.show()

# Données à visualiser
valeurs = [freq_0,2896-freq_0]  # Exemple de valeurs pour l'histogramme

# Création de l'histogramme
plt.bar(['valeur = 0','valeur existante'],valeurs, color='lightblue', edgecolor='black')

# Ajout de titres et de labels
plt.title('Caractéristique Odometre')

# Affichage de l'histogramme
plt.show()

# #### <li>La valeur 0 peut être due à un problème d'extraction des données provenant des bus CAN, donc nous allons la rejeter pour le moment. </li>

gp2_carburant = gp2_carburant.drop(['odometre'],axis =1)
gp2_carburant.head()

# ####  <font color=blue>Caractéristiques : CoutT et CoutL et id </font>

gp2_carburant = gp2_carburant.drop(['coutT','coutL','id'],axis =1)
gp2_carburant.head()

# ####  <font color=blue>Caractéristique : Note </font>

# * Cette variable contient le type de carburant versée dans une transaction (gazoil, Essence...)
# * Donc, pour une meilleure présentation et compréhension de cette variable dans ce qui suit, nous allons la renommer en **'type_carburant'** 

# renommer la colonne note en type_carburant
gp2_carburant = gp2_carburant.rename(columns={'note': 'type_carburant'})

print("Nombre d'observation par type de carburant \n \n",gp2_carburant['type_carburant'].value_counts())

#gp2_carburant

# * Nour remarquons qu'il existe le même type de carburant mais avec une différente notation de 'Gazoil' et 'Gasoil' 
# * Nous allons donc considérer que ces deux notations donnent le même type de carburant
# * Changement à faire : remplacer 'Gasoil' par 'Gazoil'

#fusionner
sub_gasoil = gp2_carburant[gp2_carburant['type_carburant'] == 'Gasoil']
sub_gasoil

gp2_carburant[gp2_carburant['type_carburant'] == 'Gazoil']

#fusionner 
gp2_carburant['type_carburant'].replace('Gasoil', 'Gazoil', inplace=True)

#(gp2_carburant[gp2_carburant['type_carburant'] == 'Gasoil']['type_carburant'] = 'Gazoil')

#juste vérification
(gp2_carburant['type_carburant']  == 'Gasoil').sum()

import matplotlib.pyplot as plt
import numpy as np

explode = (0, 0, 0.2, 0, 0)
gp2_carburant.groupby('type_carburant').size().plot(kind='pie', autopct='%1.0f%%', startangle=90,explode=explode   )
plt.title('Répartition des types de carbruant', fontsize=14)
plt.show()

# #### Exemple de fraude potentiel dans le type de carbruant utilisé par une véhicule

df_fraude = gp2_carburant[gp2_carburant['matricule'] == 932]
explode = (0.2, 0)  # on isole seulement la 4eme part 

df_fraude.groupby('type_carburant').size().plot(kind='pie', autopct='%1.0f%%', shadow=True, startangle=90,explode=explode)
plt.title('Types de carbruant : véhicule 932', fontsize=10)
plt.show()

# #### <li> On remarque que cette véhicule à utiliser parfois le Gazoil 50 ce qui peut mener à une utilisation frauduleuse de carburant pour des usages personnel par exemple. </li>

# ####  <font color=blue>Caractéristique : Station </font>

test = gp2_carburant[gp2_carburant['matricule'] == 218]
gp2_carburant['station'].unique()
explode = (0, 0, 0, 0.2, 0,0)  # on isole seulement la 4eme part 
test.groupby('station').size().plot(kind='pie', autopct='%1.0f%%', shadow=True, startangle=90)
plt.title('Stations de carburant pour mat : 218', fontsize=14)
plt.show()

# ####  <font color=blue>Caractéristique : Matricule </font>

nbr = gp2_carburant['matricule'].unique()
print("Nous avons ",len(nbr),"véhicule distincts")

#nombre d'observation par véhicule
gp2_carburant['matricule'].value_counts()

# #### La valeur de matricule 111111111  réellement signifie que la matricule de la voiture n'a pas pu être extraite/capturée
# #### Donc, pour pallier à ce problème, ....................

#Regroupement des matricules inconnues

data = gp2_carburant
from random import choice
matricules = data['matricule'].unique()
df_matricule = data[data['matricule'] == 111111111]
df_matricule
new_data = pd.DataFrame()

for numcarte in df_matricule['typePaiement'].unique() : 
    data_aux = df_matricule[df_matricule['typePaiement'] == numcarte]
    newmat = choice([i for i in range(100,1000) if i not in matricules])   
    data_aux['matricule'] = newmat
    new_data = pd.concat([new_data, data_aux], ignore_index=True)
new_data

# maintenant remplacer la partie transformée dans le dataset original 
data = data[data['matricule'] != 111111111]
data = pd.concat([data, new_data], ignore_index=True)

#avant les transformation
print("Nombre de véhicules avant la transformation : ",len(gp2_carburant['matricule'].unique()))
#après la transformation 
print("Nombre de véhicules après la transformation : ",len(data['matricule'].unique()))

gp2_carburant = data

gp2_carburant

# #### <font color = 'blue'>Trouver la différence en jours entre les date consécutives - diff_date</font>
# 
# * Pour mieux exploiter la variable date, nous allons calculer les differences en jours pour chaque 2 trasactions consécutives pour chaque véhicule à part dans une nouvelle colonne **'diff_date'**.
# 
# * Cette nouvelle colonne calculée va nous permettre de détecter les transactions qui sont réalisées dns une très courte période ou le contraire.
# 
# **<u><font color ='orange'> Exemple** </font></u>
# 
# Nous pourrons détecter si un véhicule a réalisé plusieurs transactions dans le même jours, ou bien dans une période qui différe considérablement de sa  période fréquente ou période habituelle.

# ### <font color ='red'> Ici, il faut synchroniser la variable dataframe pour grouper le résultat final dans un seul dataframe</font>
# **Mettre gp2_carburant au lieu de gp2**

gp2 = gp2_carburant

#convertir en datetime la varibale date
gp2['date'] = pd.to_datetime(gp2['date'], format='%d/%m/%Y %H:%M').dt.date
gp2['date'] 

# Trier les données par matricule puis par date (au cas où ce n'est pas déjà fait)
gp2 = gp2.sort_values(by=['matricule', 'date'])

# Calculer la différence de date entre chaque transaction et la suivante pour chaque matricule
#gp2['difference_de_date'] = gp2.groupby('matricule')['date'].diff().dt.total_seconds()
#gp2 = gp2.dropna(subset=['difference_de_date'])

gp2.head()

# Trouver la différence la plus fréquente
all_matricules = gp2['matricule']
all_matricules = all_matricules.unique()
new_dataframe = pd.DataFrame()

for mat in (all_matricules) :
    sub_data_frame = gp2[gp2['matricule'] == mat]
    # Trier les dates
    #sub_data_frame['date'] = pd.to_datetime(sub_data_frame['date'])
    dates = sub_data_frame['date']
    dates = dates.sort_values()
    # Calculer les différences entre les dates consécutives en jours
    differences = dates.diff().dt.days.dropna()
    # Trouver la différence la plus fréquente
    most_frequent_period = differences.mode()[0]
    print(most_frequent_period)
    sub_data_frame = sub_data_frame.assign(diff_date=differences)
    #recuperer la première ligne
    first_obs = sub_data_frame[0:1]
    print("before : ",sub_data_frame['diff_date'][0:1])
    sub_data_frame['diff_date'][0:1] = most_frequent_period
    print("After : ",sub_data_frame['diff_date'][0:1])
    print(sub_data_frame[0:1])
    new_dataframe = pd.concat([new_dataframe, sub_data_frame], ignore_index = True)
    
gp2_carburant = new_dataframe

# #### <font color = 'blue'> La variable typePaiment</font>
 
# * Cette carctéristique représente le numéro de carte de carbruant utilisée dans une transaction de carburant.
# * L'utilité de cette caractéristique est représenté dans le fait que chaque véhicule possède son propre carte de carburant => Tout autre comportement (utilier une autre carte) est considéré comme frauduleux.
# * Nous allons extraire juste le numéro de la carte (valeur numérique) 

#la variable typePaiment
import re
def extraire_nombres(chaine):
    nombres = re.findall(r'\d+', chaine)  # Utilisation d'une expression régulière pour extraire les nombres
    return [int(nombre) for nombre in nombres]  # Convertir les nombres en entiers
gp2_carburant['typePaiement'] = gp2_carburant['typePaiement'].apply(lambda x: extraire_nombres(x))
gp2_carburant['typePaiement'] = gp2_carburant['typePaiement'].apply(lambda x: x[0] if len(x) > 0 else None)

# ### <font color = "brown">Encodage des variables catégorielles </font>

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
gp2_carburant['station'] = le.fit_transform(gp2_carburant['station'])
gp2_carburant['type_carburant'] = le.fit_transform(gp2_carburant['type_carburant'])
gp2_carburant['fournisseur'] = le.fit_transform(gp2_carburant['fournisseur'])
#enregitrer le résultat final
gp2_carburant.to_csv('gp2_carburant_transformed_diff_date.csv', index=False)



# ## <font color ='red'>----------------------- 3 : Isolation Forest ----------------------------------------------</font>

# In[145]:
#%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# In[146]:
dataframe = pd.read_csv('/Users/abdelwahed/projet-pfe-detecteur-fraude-carburant/gp2_carburant_transformed_diff_date.csv',delimiter=',')
dataframe.head()

#dataframe['date'] = pd.to_datetime(dataframe['date'], format='%d/%m/%Y %H:%M')
#dataframe["date"] = dataframe["date"].apply(lambda x: x.timestamp())
#dataframe['date'] = dataframe['date'].dt.date
dataframe
#dataframe['date'] = dataframe['date'].sort_values()
# In[150]:

from sklearn.model_selection import cross_val_score
def train_isolation_forest_model(mat_dataset):
    
    data = mat_dataset[['matricule','quantite','type_carburant','typePaiement','fournisseur','diff_date']]
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    # Isolation forest model
    outliers_fraction = 0.2
    random_state = np.random.RandomState(42)
    ifo_model = IsolationForest(n_estimators=100,contamination='auto', max_features = 6)

    # Appliquer la validation croisée
    #scores = cross_val_score(ifo_model, data, cv=2, scoring='accuracy')
    ifo_model.fit(data)
    mat_dataset['anomaly_if'] = ifo_model.predict(data)
    return mat_dataset,ifo_model
# In[151]:
#récuperer la liste des matricules
matricules = dataframe['matricule'].unique()
len(matricules)

# In[152]:
if_final = pd.DataFrame()
print(if_final)
m = 0 
for mat in matricules :
    data_per_mat = dataframe[dataframe['matricule'] == mat]
    if_dataset,m = train_isolation_forest_model(data_per_mat)
    if_final = pd.concat([if_final, if_dataset], ignore_index = True)
    
print(mat)
print(m)    
print(if_final.shape)

# In[153]:
#enregistrer le dataset obtenu par isolation forest
if_final.to_csv('if_dataset.csv', index=False)

#tester le model sur de nouvelles données
my_saved_model = m
transactions = pd.DataFrame({
    'matricule': [837710, 837710, 837710],
    'quantite': [33.1, 35.0,33.1],
    'note': [4,4,4],
    'fournisseur': [0,0,0],
    'typePaiement': [1407058655,1407058655,1407058655],
    'diff_date': [4.0,3.0,3.0]  # Différence de date en jours
})
# In[155]:
transactions

# In[156]:
transactions['anomaly_if'] = my_saved_model.predict(transactions)

# In[157]:
transactions

# In[158]:
#afficher un exemple de données pour une matricule spécifique
res = if_final[(if_final['matricule'] == 230) & (if_final['anomaly_if'] == -1)]
res_1 = if_final[(if_final['matricule'] == 230) ]
res
# ### Visulisation de résultat d'une matricule

# In[159]:


# Afficher les résultats
fig, ax = plt.subplots(figsize = (10, 4))
anomalies = res_1[res_1['anomaly_if'] == -1]
plt.plot(res_1['date'], res_1['quantite'], label='Données originales', color='blue')

plt.scatter(anomalies['date'], anomalies['quantite'], color='red', label='Anomalies détectées', marker='o')

for index, row in anomalies.iterrows():
     plt.annotate((str(row['diff_date']) + ' jours'), (row['date'], row['quantite']), textcoords="offset points", xytext=(10,7), ha='center', fontsize=10,
                  )

plt.xlabel('Date')
plt.ylabel('Quantité de carburant versée en L')
plt.title('Visualisation des données avec les anomalies détectées')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[160]:


from sklearn.metrics import silhouette_score

# Calcul de la métrique silhouette
res_2 = res_1.drop(['date'],axis = 1)

silhouette_avg = silhouette_score(res_2, res_1['anomaly_if'])
print("Silhouette Score:", silhouette_avg)

# In[161]:
data_per_mat = if_final[if_final['matricule'] == 230]
data_per_mat_1= data_per_mat.drop(['date','anomaly_if','station'],axis =1)
data_per_mat['scores'] = my_saved_model.decision_function(data_per_mat_1)
data_per_mat['scores']
# In[162]:

import matplotlib.pyplot as plt

# Tracer un histogramme des scores de détection d'anomalies

plt.hist(data_per_mat['scores'], bins=50, color='blue', alpha=0.7)
plt.xlabel('Score de détection d\'anomalies')
plt.ylabel('Fréquence')
plt.title('Distribution des scores de détection d\'anomalies')
plt.show()


# In[163]:


import numpy as np

# Calcul de la stabilité en utilisant les prédictions de l'Isolation Forest
def compute_stability(predictions, num_runs=20):
    n_samples = len(predictions)
    stability_scores = []
    for _ in range(num_runs):
        random_predictions = np.random.choice(predictions, n_samples, replace=True)
        stability_score = np.mean(predictions == random_predictions)
        stability_scores.append(stability_score)
    return np.mean(stability_scores)

# Supposons que anomaly_predictions sont les prédictions de votre modèle Isolation Forest

stability_score = compute_stability(res_1['anomaly_if'])
print("Stability Score:", stability_score)

# In[167]:
import seaborn as sns
import pandas as pd
# Tracez un pairplot pour visualiser les relations entre les caractéristiques
res_2 = res_1.drop(['matricule','anomaly_if'],axis = 1)
sns.pairplot(res_2)
plt.show()

# ### <font color = 'orange'> ------------Visualisation Interactive-------------------- </font>
# In[168]:
import plotly.io as pio
pio.renderers.default = "notebook_connected"
# In[171]:
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Ajouter la trace principale pour les données
fig = px.line(res_1, x='date', y='quantite',title='Visualisation des données avec les anomalies détectées',
              labels={'quantite': 'Quantité'}, 
              hover_data=['type_carburant','diff_date'])

# Ajouter une trace pour les anomalies
fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['quantite'], mode='markers', name='Anomalies',
                         marker=dict(color='red', size=10),
                         hoverinfo='text', # Utiliser 'text' pour que seules les informations personnalisées soient affichées
                         hovertext=anomalies.apply(lambda row: f"Date : {row['date']}<br>Quantité : {row['quantite']}<br>Type_carburant : {row['type_carburant']}<br>Diff_period : {row['diff_date']} jours", axis=1)))
  
# Amélioration du formatage
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Quantité de carburant versée en L',
    legend_title='Légende',
    xaxis_tickangle=-25
)
# Affichage du graphique
fig.show()

# ### <font color ='green'>Résultat final obtenu grâce à Isolation Forest</font>

# In[172]:
#pie chart plot
plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)

explode = (0, 0.2)  # on isole seulement la 1ère part 
#add colors
colors = ['#D93F15','#67966A']

if_final.groupby('anomaly_if').size().plot(kind='pie', autopct='%1.0f%%',colors =colors, shadow=True, 
                                           startangle=90, explode= explode,labels = ['Transactions \n anormales','Transactions \n Normales'])
plt.title('Pie Chart visualisation', fontsize=10)

# bar plot
plt.subplot(1,2,2)
bars = if_final.groupby('anomaly_if').size().plot(kind='bar',color =['#D9CA15','#030D8C'])
plt.title('Bar Plot visualisation', fontsize=10)

#afficher les figures
plt.suptitle('Proportion globale des anomalies',fontsize=14,fontweight ='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste les subplots dans la figure pour laisser de l'espace pour le suptitle
plt.show()


# ## <font color ='blue'>-------------------- Explainable AI for Isolation Forest -----------------------------</font>

# In[88]:
#récuperer les données d'une matricule spécifique
sub_mat = dataframe[dataframe['matricule'] == 932]
#print(sub_mat)

# Filtrer les données
sub_mat = sub_mat[['matricule','quantite','type_carburant','typePaiement','fournisseur','diff_date']]
#print(sub_mat)

dataset_test, sub_model = train_isolation_forest_model(sub_mat)
print(dataset_test.head())
#feature_names = ['quantite','note','typePaiement','fournisseur','diff_date']
feature_names = sub_mat.columns
# In[848]:
#pip install lime

# In[89]:
#pip install shap
import shap
shap.initjs()

# In[90]:
import shap
import matplotlib.pyplot as plt

observation_index = 29
#supprimer le target du sub_datarame
#sub_mat = sub_mat.drop(['anomaly_if'],axis=1)

explainer = shap.TreeExplainer(sub_model, sub_mat)
print(sub_model)
print(feature_names)

# Calculate SHAP values
shap_values1 = explainer.shap_values(sub_mat.iloc[observation_index,:])

shap_values = explainer.shap_values(sub_mat)

# Plot the SHAP values
shap.force_plot(explainer.expected_value,shap_values1, sub_mat.iloc[observation_index,:], feature_names=feature_names)

#display(p)

# ### <font color ='maroon'>Model Global Explanation - SHAP</font>

# In[91]:
shap.summary_plot(shap_values,sub_mat,feature_names=feature_names) 
# In[112]:
#shap.plots.waterfall(shap_values[0])


# ###  <font color = 'maroon'>LIME - Explication locale de résultat de prédiction de Isolation Forest</font>
# 
# * Algorithme fournit des explications interprétables pour les prédictions individuelles faites par les modèles d'apprentissage automatique.
# 
# * Fonctionemment : crée une explication locale autour de la prévision plutot que d'essayer de comprendre l'ensemble du modèle. 
# 
# * LIME génère une explication en entraînant un modèle simple interprétable sur un petit sous-ensemble de points de données proches de la prédiction d'intérêt plutôt que d'essayer de comprendre le modèle plus complexe et applicable à l'échelle mondiale.
# 
# * Cela permet des explications compréhensibles par l'homme et indépendantes du modèle, ce qui signifie qu'elles peuvent être appliquées à n'importe quel modèle d'apprentissage automatique, quelle que soit sa structure interne.

# In[117]:
from lime.lime_tabular import LimeTabularExplainer

#test_sub_mat_for_lime = dataframe[dataframe['matricule'] == 932]
test_sub_mat_for_lime = sub_mat

#y_train = sub_mat['anomaly_if']
test_sub_mat_for_lime = test_sub_mat_for_lime.drop(['anomaly_if'],axis=1)
# Obtenir les scores d'anomalie (plus le score est bas, plus la probabilité d'être une anomalie est élevée)
scores = sub_model.decision_function(test_sub_mat_for_lime)

feature_names = test_sub_mat_for_lime.columns.tolist()  # Obtenez les noms des caractéristiques

# Créez un explainer LIME
explainer = LimeTabularExplainer(test_sub_mat_for_lime.values, mode="regression", 
                                 training_labels=scores,feature_names=feature_names,
                                class_names=[-1, 1])

# Choisissez un échantillon d'anomalie
sample_index = 42
sample = test_sub_mat_for_lime.iloc[1]

# Obtenez l'explication LIME
explanation = explainer.explain_instance(sample, sub_model.decision_function,num_features=6)
explanation.show_in_notebook(show_all=True)  # Affiche l'explication dans un notebook (vous pouvez également l'enregistrer dans un fichier)

# **Dans ce cas, les poids retournés par LIME peuvent être interprétés comme suit :**
# 
# * Valeur positive d'une caractéristique : Augmenter cette caractéristique rend l'instance plus "normale" selon le modèle.
# 
# * Valeur négative d'une caractéristique : Augmenter cette caractéristique rend l'instance plus susceptible d'être une anomalie.
# 

# In[118]:
# Fermer toutes les figures ouvertes
plt.close('all')

explanation.as_pyplot_figure()
plt.show()

# #### Interprétation : 
# Ce graphique nous montre que la variable note (type de carburant) est la cause principale de cette anomalie détecté

# In[119]:
important_features = explanation.as_list()
# Afficher les caractéristiques importantes
print("Caractéristiques importantes pour la prédiction de l'anomalie : \n \n")
for feature in important_features:
    feature_name = feature[0]  # Nom de la caractéristique
    feature_weight = feature[1]  # Poids de la caractéristique dans la prédiction
    print(f"- {feature_name}: {feature_weight}")


# #### Le feature qui a affecté le plus le résultat de sub modèle de IF

# In[120]:
import numpy
#print(important_features)
most_contributed_feature = min(important_features, key=lambda x: x[1])

#most_contributed_feature = min(important_features[1])
print(most_contributed_feature)