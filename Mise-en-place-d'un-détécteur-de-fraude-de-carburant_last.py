#!/usr/bin/env python
# coding: utf-8

# # Mise en place d'un détecteur de fraude de carburant

# ## <font color='blue'> ------------------------------------**Données des bus CAN**---------------------------------------</font>
# 

# ## <font color='orane'> **1 - Chargement et affichage des informations de base du jeu de données :**</font>

# ### **1.1 - Les premières ligne des jeux données rep_carburant et gp2_carburant**

# In[38]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# ### I- Analyse, compréhension et prétraitement de données

# In[39]:


# Penser à des idées pour la visualisation de donées :
# des preuves visuales pour prouver l'incorrespondance des données 
import pandas as pd


# In[115]:


gp2_carburant = pd.read_csv('/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/data/input-data/gp2_carburant.csv',delimiter=';',encoding='utf-8')
gp2_carburant.sample(5)


# In[41]:


gp2_carburant.shape


# <li> Taille de jeu de données : 7935 enregistrements et 18 caractéristiques </li>

# ### Liste de noms de colonnes 

# In[42]:


colonnes = gp2_carburant.columns
nbrcolonnes = len(gp2_carburant.columns)
print('Liste des caractéristiques dans le jeu de données \n', colonnes)
print('Nombre de caractéristiques dans le jeu de données : ', nbrcolonnes)


# ### Vérification des données manquantes

# In[43]:


print(gp2_carburant.isnull().sum())


# ### Analyse des caractéristiques

# In[44]:


cols = ['is_modified_qty','is_modified_date','is_modified_odometre','is_modified_coutL','lon','lat','numPaiement']
for col in cols :
    print("Les valeurs uniques de la colonne : ",col," : ",gp2_carburant[col].unique())


# #### <li> Toutes les valeurs de ces 7 colonnes sont des 0 ou NULL, donc nous allons les suprrimer</li> 

# In[45]:


gp2_carburant = gp2_carburant.drop(['is_modified_qty','is_modified_date','is_modified_odometre','is_modified_coutL','lon','lat','numPaiement'],axis = 1)
#Vérifions la suppresion 
gp2_carburant.head()


# ####  <font color=blue>**Caractéristique : Odometre** </font>
# 

# In[46]:


import matplotlib.pyplot as plt
import numpy as np

# La colonne odometre
freq_0 = gp2_carburant['odometre'].value_counts()[0]
freq = freq_0 * 100 / 7935
print(freq_0,"enregistrements avec 0 comme valeur dans la colonne Odometre avec un pourcentage de : ",freq,"%")

couleurs = ['lightgreen', 'lightgrey']
plt.pie([100-freq,freq], autopct='%1.0f%%', colors = couleurs,labels = ['valeus existantes', 'O comme valeur']
)

plt.title('Fréquence de la valeur 0 dans Odometre', fontsize=12)

plt.show()


# In[47]:


# Données à visualiser
valeurs = [freq_0,7935-freq_0]  # Exemple de valeurs pour l'histogramme

# Création de l'histogramme
plt.bar(['valeur = 0','valeur existante'],valeurs, color='lightblue', edgecolor='black')

# Ajout de titres et de labels
plt.title('Caractéristique Odometre')

# Affichage de l'histogramme
plt.show()


# #### <li>La valeur 0 peut être due à un problème d'extraction des données provenant des bus CAN, donc nous allons la rejeter pour le moment. </li>

# In[48]:


gp2_carburant = gp2_carburant.drop(['odometre'],axis =1)
gp2_carburant.head()


# ####  <font color=blue>Caractéristiques : CoutT et CoutL et id </font>

# In[49]:


gp2_carburant = gp2_carburant.drop(['coutT','coutL','id'],axis =1)
gp2_carburant.head()


# ####  <font color=blue>Caractéristique : Note (type_carburant) </font>

# * Cette variable contient le type de carburant versée dans une transaction (gazoil, Essence...)
# * Donc, pour une meilleure présentation et compréhension de cette variable dans ce qui suit, nous allons la renommer en **'type_carburant'** 

# In[50]:


# renommer la colonne note en type_carburant
gp2_carburant = gp2_carburant.rename(columns={'note': 'type_carburant'})


# In[51]:


print("Nombre d'observation par type de carburant \n \n",gp2_carburant['type_carburant'].value_counts())


# In[52]:


#gp2_carburant


# * Nour remarquons qu'il existe le même type de carburant mais avec une différente notation de 'Gazoil' et 'Gasoil' 
# * Nous allons donc considérer que ces deux notations référencent le même type de carburant.
# * Changement à faire : remplacer 'Gasoil' par 'Gazoil'

# In[53]:


#fusionner
sub_gasoil = gp2_carburant[gp2_carburant['type_carburant'] == 'Gasoil']
sub_gasoil


# In[54]:


gp2_carburant[gp2_carburant['type_carburant'] == 'Gazoil']


# In[55]:


#fusionner 
gp2_carburant['type_carburant'].replace('Gasoil', 'Gazoil', inplace=True)

#(gp2_carburant[gp2_carburant['type_carburant'] == 'Gasoil']['type_carburant'] = 'Gazoil')


# In[56]:


#juste vérification
(gp2_carburant['type_carburant']  == 'Gasoil').sum()


# In[57]:


import matplotlib.pyplot as plt
import numpy as np

explode = ( 0, 0, 0, 0,0)

gp2_carburant.groupby('type_carburant').size().plot(kind='pie', autopct='%1.0f%%', startangle=90,explode=explode   )
plt.title('Répartition des types de carbruant', fontsize=14)

plt.show()


# #### Exemple de fraude potentiel dans le type de carbruant utilisé par une véhicule
# 

# In[58]:


df_fraude = gp2_carburant[gp2_carburant['matricule'] == 932]
explode = (0.2, 0)  # on isole seulement la 4eme part 

df_fraude.groupby('type_carburant').size().plot(kind='pie', autopct='%1.0f%%', shadow=True, startangle=90,explode = explode)
plt.title('Types de carbruant : véhicule 932', fontsize=10)

plt.show()


# #### <li> On remarque que cette véhicule à utiliser parfois le Gazoil 50 ce qui peut mener à une utilisation frauduleuse de carburant pour des usages personnel par exemple. </li>

# ####  <font color=blue>Caractéristique : Matricule </font>
# 

# In[59]:


nbr = gp2_carburant['matricule'].unique()
print("Nous avons ",len(nbr),"véhicules distincts")


# In[60]:


#nombre d'observation par véhicule
gp2_carburant['matricule'].value_counts()


# #### La valeur de matricule 111111111  réellement signifie que la matricule de la voiture n'a pas pu être extraite/capturée
# #### Donc, pour résoudre ce problème, nous allons les regrouper par le numéro de la carte carbruant en lui attribuant les valeurs de matricule aléaatoires.

# In[61]:


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


# In[62]:


# maintenant remplacer la partie transformée dans le dataset original 
data = data[data['matricule'] != 111111111]
data = pd.concat([data, new_data], ignore_index=True)


# In[63]:


#avant les transformation
print("Nombre de véhicules avant la transformation : ",len(gp2_carburant['matricule'].unique()))
#après la transformation 
print("Nombre de véhicules après la transformation : ",len(data['matricule'].unique()))


# In[64]:


gp2_carburant = data


# In[65]:


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

# In[66]:


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


# In[67]:


# Trouver la différence la plus fréquente
all_matricules = gp2['matricule']
all_matricules = all_matricules.unique()
new_dataframe = pd.DataFrame()


# In[68]:


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
    


# In[69]:


gp2_carburant = new_dataframe


# In[70]:


gp2_carburant.isnull().sum()


# #### <font color = 'blue'> La variable typePaiment (Numéro de carte Carburant)</font>
# 
# *  Cette carctéristique représente le numéro de carte de carbruant utilisée dans une transaction de carburant.
# 
# 
# *  L'utilité de cette caractéristique est représenté dans le fait que chaque véhicule possède son propre carte de carburant => Tout autre comportement (utilier une autre carte) est considéré comme frauduleux.
# 
# 
# *  Cette variable contient une valeur manquantes (déjà vérifiée au début de l'analyse ci-dessus) 
# 
# 
# *  Pour l'imputer, nous allons nous procéder comme suit :
# 
# 
#       - Chercher la transaction qui contient la valur manquante dans typePaiement( numéro de carte carburant.
#       - Imputer cette dernière avec la valeur médiane de cette colonne.

# In[77]:


len(gp2_carburant['typePaiement'].unique())


# In[78]:


# Identifier les lignes avec des valeurs manquantes
ligne_manquante = gp2_carburant[gp2_carburant.isnull().any(axis=1)]
ligne_manquante


# In[79]:


temp_data_per_mat = gp2_carburant[gp2_carburant['matricule'] == 1624]
temp_data_per_mat


# In[80]:


median_value = temp_data_per_mat['typePaiement'].mode()[0]
median_value


# In[81]:


#imputer maintenant la valeur manquante
ligne_manquante['typePaiement'] = median_value
gp2_carburant.iloc[ligne_manquante.index] = ligne_manquante 
gp2_carburant.iloc[3073]


# * Nous allons extraire juste le numéro de la carte (valeur numérique) 

# In[82]:


from sklearn.preprocessing import LabelEncoder

# Créer une instance de LabelEncoder
label_encoder_tp = LabelEncoder()

# Adapter l'encodeur aux données d'entraînement
label_encoder_tp.fit(gp2_carburant['typePaiement'])  # Remplacez 'variable_catégorielle' par le nom de votre colonne catégorielle

# Transformer les données d'entraînement
gp2_carburant['typePaiement_enc'] = label_encoder_tp.transform(gp2_carburant['typePaiement'])


# In[83]:


gp2_carburant['typePaiement']


# In[84]:


#la variable typePaiment
#import re
#def extraire_nombres(chaine):
#    nombres = re.findall(r'\d+', chaine)  # Utilisation d'une expression régulière pour extraire les nombres
#    return [int(nombre) for nombre in nombres]  # Convertir les nombres en entiers
#gp2_carburant['typePaiement'] = gp2_carburant['typePaiement'].apply(lambda x: extraire_nombres(x))
#gp2_carburant['typePaiement'] = gp2_carburant['typePaiement'].apply(lambda x: x[0] if len(x) > 0 else None)

#gp2_carburant


# gp2_carburant.head()

# ####  <font color=blue>Caractéristique : Station </font>
# 

# In[85]:


test = gp2_carburant[gp2_carburant['matricule'] == 218]
#gp2_carburant['station'].unique()


# In[86]:


explode = (0.2, 0, 0, 0)  # on isole seulement la 4eme part 

test.groupby('station').size().plot(kind='pie', autopct='%1.0f%%', shadow=True, startangle=90)
plt.title('Stations de carburant pour mat : 218', fontsize=14)

plt.show()


# #### Pour une bonne détection d'anomalies  si elle existe, nous avons décidé de regoruper les stations de service SHELL par gouvernorat.
# #### Le traitement de regrouepent des stations par gouvernorat est fait dans un fichier à part pour ne pas charger le notebook.
# #### Dans ce qui suit, nous allons importer le nouveau fichier de données après le traitement réaliser pour continuer le reste du precessus de prétraitement. 

# In[87]:


gp2_carburant.to_csv('gp2_carburant_transformed_diff_date.csv', index=False)


# ### <font color = "brown">Encodage des variables catégorielles </font>

# #### Nous allons importer le fichier qui contient les gouvernorats obtenu à partir du web scrapping, et puis nous allons les encoder pour l'entrainement des modèles dans la partie Implémentation

# In[107]:


from sklearn.preprocessing import LabelEncoder

gp2_carburant = pd.read_csv('gp2_carburant_transformed_diff_date_2.csv',delimiter =",")
gp2_carburant.head()


# In[110]:


le_station = LabelEncoder()
le_carburant = LabelEncoder()
le_fournisseur = LabelEncoder()
gp2_carburant['station_enc'] = le_station.fit_transform(gp2_carburant['station'])
gp2_carburant['type_carburant_enc'] = le_carburant.fit_transform(gp2_carburant['type_carburant'])
gp2_carburant['fournisseur_enc'] = le_fournisseur.fit_transform(gp2_carburant['fournisseur'])
gp2_carburant['gouvernorat_enc'] = le_fournisseur.fit_transform(gp2_carburant['gouvernorat'])


# In[114]:


#gp2_carburant.head()
x = gp2_carburant[gp2_carburant['matricule'] == 230]
x['gouvernorat_enc'].unique()


# In[109]:


gp2_carburant['gouvernorat'].unique()


# In[96]:


gp2_carburant.to_csv('gp2_carburant_transformed_diff_date_3.csv', index=False)


#  ## --------------- Analyse des Series temporelles : Simple Exponential Smoothing ----------------
# 

# 
# **Quand utiliser ?**
# 
#  - Peu de points de données, 
#  
#  - Données irrégulières.
# 
#  - Aucune saisonnalité ni tendance.
#  
# **Paramètres**
# 
# * **alpha** : pour contrôler la rapidité avec laquelle les influences des observations passées décroissent.
# 

# In[924]:


df3 = pd.read_csv('/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date.csv',delimiter =",")

df3['date'] = pd.to_datetime(df3['date'])
# Définir la colonne 'Date' comme index
df3.set_index('date', inplace=True)
df = df3[df3['matricule'] == 230]
df = df[['quantite']]


# In[925]:


len(df)


# In[926]:


#plt.style.use('seaborn-darkgrid')
plt.figure(figsize = (12, 6))
#plt.xlabel('Date', fontsize = 14)
plt.ylabel('Quantité', fontsize = 10)
#df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
plt.plot(df.index, df['quantite'], label = 'quantité')
plt.xticks(rotation=45)
plt.title('Achat de carburant par mois pour matricule 230', fontsize = 13)
plt.legend() 
plt.show() 


# #### 1. Test de stationnarité : les propriétés statistiques, comme la moyenne, la variance et l'autocorrélation, restent constantes dans le temps. 
# * On procède ensuite à un test de stationnarité. 
# 
# * Le test de **Dickey-Fuller** donne de bons résultats rapidement.
# 
# * Ce score indique si la série peut être considérée comme stationnaire ou non. 
# 
# * En général, s’il est inférieur à 0.05, on la considère stationnaire.

# In[166]:


from statsmodels.tsa.stattools import adfuller

_, p, _, _, _, _ = adfuller(df)
print("La p-value est de: ", round(p, 3))


# 
# 
# 
# #### Interprétation 
# 
# * La série temporelle n'est pas sationnaire ca la p-value est > 0.5
# * On ne peut pas modéliser donc notre série temporelle avec ARIMA (modèle prédictif)

# #### 2. Décomposotion et analye de la série temporelle

# In[167]:


import pandas as pd
import statsmodels.api as sm

# Décomposer la série temporelle
decomposition = sm.tsa.seasonal_decompose(df['quantite'], model='additive',period=10)

# Afficher les composantes de la décomposition
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# Tracer les composantes de la décomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.xticks(rotation=45)

plt.plot(df, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.xticks(rotation=45)

plt.plot(trend, label='Tendance')
plt.legend(loc='best')
plt.subplot(413)
plt.xticks(rotation=45)

plt.plot(seasonal, label='Saisonnalité')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Résidus')
plt.legend(loc='best')
plt.xticks(rotation=45)

plt.tight_layout()


# #### Interprétation
# Mana3rash shnoa shen9oul hné
# comment interpréter la tendance et la saisonalité
# 

# #### Simple Exponetial Smoothing - Implémentation

# In[168]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

#df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
#df.set_index(df['date'])

#model = ExponentialSmoothing(df['quantite'], seasonal='add', seasonal_periods=4)  # Par exemple, avec une saisonnalité hebdomadaire
#model._index = pd.to_datetime(df['date

# Créez le modèle de lissage exponentiel
model = SimpleExpSmoothing(df['quantite']) 

# Ajustez le modèle aux données
fit_model = model.fit(smoothing_level=.8)

# Prévoyez les valeurs
df['SES_Predictions'] = fit_model.fittedvalues

#pred1 = fit_model.forecast(5)
#print(pred1)
predictions = fit_model.predict(start=0, end=len(df)-1)


# In[169]:


predictions
#pred1


# In[131]:


# Calcul des résidus
df['Residuals'] = df['quantite'] - df['SES_Predictions']

# Détection d'anomalies où l'écart est supérieur à deux fois l'écart-type des résidus
std_dev = np.std(df['Residuals'])
df['Anomaly'] = df['Residuals'].apply(lambda x: 'Anomaly' if np.abs(x) > 1 * std_dev else 'Normal')

# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(df['quantite'], label='Original Data')
plt.plot(df['SES_Predictions'], label='SES Predictions', color='red')
plt.scatter(df.index, df['quantite'], c=df['Anomaly'].apply(lambda x: 'red' if x == 'Anomaly' else 'blue'), label='Anomalies if red')
plt.title('Simple Exponential Smoothing and Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[170]:


# Calculez la moyenne et l'écart-type de la série temporelle
mean = df['quantite'].mean()
std = df['quantite'].std()

# Définissez un seuil en fonction de l'écart-type
thresholdmax = mean + 0.8 * std 
thresholdmin = mean - 0.8 * std  # Par exemple, définir le seuil à 3 écarts-types au-dessus de la moyenne
print('le seuil max = ',thresholdmax)
print('le seuil min = ',thresholdmin)
# Identifiez les anomalies
anomalies = df[df['quantite'] >= thresholdmax] 
anomalies = pd.concat([anomalies,df[df['quantite'] <= thresholdmin]],ignore_index=True)
#or df[df['quantite'] < threshold]


# In[171]:


anomalies


# In[172]:


plt.figure(figsize = (12, 6))
#plt.xlabel('Date', fontsize = 14)
plt.ylabel('Quantité', fontsize = 10)
#aux = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
plt.plot(df.index, df['quantite'], label = 'quantité réelle')
plt.plot(df.index, df['SES_Predictions'], label = 'quantité prédite')

#plt.plot(df['date'], pred1, label = 'quantité prédite',color ='orange')

#plt.scatter(anomalies['date'], anomalies['quantite'],color='red', label='Anomalies détectées', marker='o')

plt.xticks(rotation=45)
plt.title('Achat de carburant pour matricule 230', fontsize = 13)
plt.legend() 
plt.show()


# #### Interprétation
# 
# * Nous remarquons que les résulats de prédiction de Simple Exponential Smoothing ne sont pas performants.
# 
# * En effet, pour de bonnes prédictions de série temorolles ,il faut utiliser de données sans anomalies, mais ce n'est pas le ca pour nous.

# # Apprentissage non supervisé pour la détection d'anomalies
# 

# ## ------------------------------ 1 : SOM : Self - Organized Maps -------------------------------------

# In[186]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import bone, pcolor, colorbar, plot, show
import pylab as pl


# In[187]:


#pip install minisom


# In[190]:


gp2_carburant = pd.read_csv("/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date.csv",delimiter =',')

gp2_carburant['date'] = pd.to_datetime(gp2_carburant['date'])
gp2_carburant.head()


# In[191]:


from mpl_toolkits.mplot3d import Axes3D

# Créer une figure et un axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data_plot = gp2_carburant[['quantite','station','matricule']]
# Extraire les coordonnées x, y et z des données
x_data = data_plot['quantite']
y_data = data_plot['station']
z_data = data_plot['matricule']

# Tracer les données
ax.scatter(x_data, y_data, z_data)

# Ajouter des étiquettes et un titre
ax.set_xlabel('quantite')
ax.set_ylabel('station')
ax.set_zlabel('matricule')
ax.set_title('Data in 3D Plot')


# In[192]:


data = gp2_carburant
#d = data
#d['Year'] = d['date'].dt.year
#d['Month'] = d['date'].dt.month
#d['Day'] = d['date'].dt.day
#d['DayOfWeek'] = d['date'].dt.dayofweek
#d['DayOfYear'] = d['date'].dt.dayofyear
#d['WeekOfYear'] = d['date'].dt.isocalendar().week
data.head()


# In[193]:



# Trier les données par matricule puis par date (au cas où ce n'est pas déjà fait)
gp2 = gp2_carburant
gp2 = gp2.sort_values(by=['matricule', 'date'])

# Calculer la différence de date entre chaque transaction et la suivante pour chaque matricule
#gp2['difference_de_date'] = gp2.groupby('matricule')['date'].diff().dt.total_seconds()
#gp2 = gp2.dropna(subset=['difference_de_date'])

gp2.head()


# #### Trouver les différences en jours entre les dates consécutives

# In[194]:


new_dataframe['diff_date'].isnull().sum()


# In[195]:


new_dataframe['diff_date']


# In[198]:


gp2 = new_dataframe

gp2_1 = gp2
#gp2_1 = gp2.drop(['Year','Month','DayOfWeek','DayOfYear','WeekOfYear','Day','station'],axis =1)
gp2 = gp2[gp2['matricule'] == 932]
gp2_1 = gp2_1[gp2_1['matricule'] == 932]

#gp2 = gp2.drop(['date','Year','Month','DayOfWeek','DayOfYear','WeekOfYear','Day','station'],axis =1)
#gp2 = gp2[['matricule','quantite']]
gp2


# In[199]:


gp2 = gp2.drop(['date','station'], axis =1)


# ### Feature Scaling

# In[200]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
gp2_scaled = sc.fit_transform(gp2)


# In[201]:


from sklearn.preprocessing import StandardScaler
 
object= StandardScaler()
  
# standardization 
gp2_scaled = object.fit_transform(gp2) 
#print(scale)


# ### Training the SOM
# 

# In[202]:


map_size = 5 * math.sqrt(len(gp2_scaled))
map_height = map_width = math.ceil(math.sqrt(map_size))


# In[203]:


print(f'(map_height, map_width) = ({map_height}, {map_width})')
print(f'Number of features: {gp2_scaled.shape[1]}')


# In[204]:


gp2_scaled.shape[1]


# In[348]:


from minisom import MiniSom
from sklearn.metrics import silhouette_score

som = MiniSom(x=map_width, y=map_height, input_len = gp2_scaled.shape[1], sigma = 1.5, learning_rate =0.1
              ,neighborhood_function='gaussian')

som.random_weights_init(gp2_scaled)

#verbose=True : demandez au modèle SOM d'afficher des informations supplémentaires pendant l'entraînement
som.train_batch(data = gp2_scaled, num_iteration = 1500, verbose = True)


# ### <font color = "blue">Évaluation</font>
# * Nous évaluons les performances des SOM en nous concentrant sur deux aspects différents. L'une est la capacité en tant que méthode de clustering, que nous étudions en utilisant l'analyse de silhouette ; l'autre préserve la topologie de l'espace d'entrée, par analyse erreur topographique. 
# * Nous sélectionnons la métrique d'évaluation en fonction de son utilisation généralisée dans l'évaluation de clustering (analyse de silhouette) et les caractéristiques du SOM

# **1 - L'erreur de quantification** : 
# 
# - Est la mesure de la différence moyenne entre les vecteurs d'entrée et leurs unités gagnantes correspondantes sur la carte SOM. 
# - Plus cette erreur est faible => meilleure est la représentation des données sur la carte SOM avec des représentations proches des données d'entrée..

# **<font color = "red">Interprétation</font>**
# * On peut dire que SOM nous a donné une bonne représentation des données sur la carte SOM avec une valeur de QE =0.066

# **2- La métrique Erreur Topographique**
# 
# * Le Topographic Error mesure la préservation de la topologie dans une carte auto-organisatrice (SOM). 
# 
# * Il évalue la proportion d’erreurs topologiques, c’est-à-dire les voisins dans l’espace d’entrée qui ne sont pas voisins dans la carte.

# In[349]:


# Calcul de l'erreur topologique
topographic_error = som.topographic_error(gp2_scaled)


# In[350]:


topographic_error


# **<font color = "red">Interprétation</font>**
# * Une valeur de 0.7 suggère que, en moyenne, 70% des voisins de chaque unité de la carte ne sont pas directement connectés à elle sur la carte. 
# 
# * Cela peut indiquer que la représentation des données sur la carte SOM n'est pas optimale et que les relations spatiales entre les données ne sont pas bien capturées.
# 
# * Une erreur topographique plus faible indique que les unités voisines sur la carte sont plus susceptibles de représenter des données similaires dans l'espace d'entrée.
# 
# * Une erreur topographique faible indique que les relations spatiales entre les données sont bien préservées, avec des données similaires placées à proximité les unes des autres sur la carte.

# **2 - La métrqiue Silouhette**
# * Cette métrique fait .......

# In[351]:


#Extrayez les étiquettes des clusters à partir des coordonnées des unités gagnantes
cluster_labels = np.array([som.winner(x) for x in gp2_scaled])
#print(cluster_labels)
# Appliquez une transformation pour obtenir un tableau unidimensionnel
labels_1d = cluster_labels[:, 0] * som._weights.shape[1] + cluster_labels[:, 1]
#print(labels_1d)

#labels = np.array([som.winner(x) for x in gp2_scaled])

# Calcul du coefficient de silhouette
silhouette_avg = silhouette_score(gp2_scaled, labels_1d)
print("Silhouette Score:", silhouette_avg)


# In[352]:


import numpy as np

weights = som.get_weights()

# Enregistrement des poids dans un fichier
np.save("som_weights.npy", weights)


# In[353]:



# Charger les poids à partir du fichier
#weights = np.load("som_weights.npy")

# Créer un nouveau modèle SOM avec les poids chargés
#new_model = MiniSom(x=map_width-1, y=map_height-1, input_len=gp2_scaled.shape[1], sigma = 1.4,learning_rate =0.1) 
#new_som.weights = weights.reshape((gp2_scaled, new_model._weights.shape[1], gp2_scaled.shape[1]))


# In[354]:


import numpy as np
import matplotlib.pyplot as plt

# Calculer la distance moyenne entre chaque unité et ses voisins
u_matrix = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

for i in range(som.get_weights().shape[0]):
    for j in range(som.get_weights().shape[1]):
        distances = []
        for ii in range(som.get_weights().shape[0]):
            for jj in range(som.get_weights().shape[1]):
                distances.append(np.linalg.norm(som.get_weights()[i, j] - som.get_weights()[ii, jj]))
        u_matrix[i, j] = np.mean(distances)

# Tracer la U-matrix
plt.imshow(u_matrix, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('U-matrix')
plt.show()


# In[355]:



# Créer un tableau pour stocker les ID attribués à chaque unité de la carte SOM
ids_par_neurone = np.empty((som.get_weights().shape[0], som.get_weights().shape[1]))

# Pour chaque échantillon dans les données
for i, x in enumerate(data):
    # Trouver l'unité gagnante pour l'échantillon
    winner_unit = som.winner(x)
    # Associer l'ID correspondant à l'unité gagnante
    ids_par_neurone[winner_unit[0], winner_unit[1]] = df.iloc[i]['ID']


# In[334]:


import matplotlib.pyplot as plt

# Créer une figure
plt.figure(figsize=(10, 8))

# Tracer la carte SOM
for i in range(som.get_weights().shape[0]):
    for j in range(som.get_weights().shape[1]):
        # Obtenir l'ID correspondant à l'unité de la carte SOM
        ID = ids_par_neurone[i, j]
        # Si l'ID existe, tracer un carré coloré avec l'ID comme couleur
        if ID is not None:
            plt.plot(j, som.get_weights()[i, j], marker='s', markersize=20, color=ID, markeredgecolor='k')

# Ajouter une légende
plt.legend()

# Afficher la carte SOM
plt.title('Carte SOM avec Couleurs Correspondant aux IDs')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()


# ### Comprendre les résultats 
# 
# Le nœud gagnant est celui qui est le plus proche de l’entité d’entrée Xi.
# 
# Nous pouvons obtenir ses coordonnées (x, y) sur la carte en utilisant la méthode winner().

# In[356]:


som.winner(gp2_scaled[0])


# La carte de distance est un tableau 2D (17x17) où chaque élément représente la distance moyenne entre un neurone et ses voisins.

# In[357]:


print('-------------\nDistance Map\n------------')
print(f'Shape: {som.distance_map().shape}')
print(f'First Line: {som.distance_map().T[0]}')


# ### Les fréquences : array 2D qui contient chaque neuronne combien de fois a gagné comme BMU
# Exemple : neuronne localisé à (2, 3) à gagné 3 fois comme BMU

# In[358]:


frequencies = som.activation_response(gp2_scaled)
print(f'Fréquences:\n {np.array(frequencies.T, np.uint)}')


# #### Visualiser les resultats

# In[359]:


# Générer la U-Matrix
u_matrix = som.distance_map()

# Visualiser la U-Matrix
plt.figure(figsize=(7, 7))
plt.pcolor(u_matrix.T, cmap='bone_r')  # Transposée pour correspondre à l'orientation habituelle
plt.colorbar(label='Distance')
plt.show()


# #### Interprétation
# 
# * Les nuances claires représentent les clusters tandis que les plus sombres représentent la séparation entre ces clusters.
# 
# * Les zones claires indiquent des groupes de neurones similaires.
# 
# * Les zones sombres peuvent révéler des séparations entre différents types de données.

# In[360]:


plt.figure(figsize=(7, 7))
fréquences = som.activation_response (gp2_scaled)
plt.pcolor(frequencies.T, cmap='Blues')
plt.colorbar()
plt.show()


# ### Identifier les anomalies

# In[361]:


# Calculer la distance à BMU pour chaque donnée
distances_to_bmu = np.array([som.winner(d) for d in gp2_scaled])

# Calculer la norme de ces distances
norm_distances = np.linalg.norm(distances_to_bmu, axis=1)
print(type(norm_distances))


# In[362]:


# Définir un seuil
seuil = np.mean(norm_distances) + 1 * np.std(norm_distances)
print('seuil : ',seuil)

# Identifier les indices des anomalies
anomalies_indices = np.where(norm_distances > seuil)

#print(anomalies_indices)
original_data_with_one_mat = gp2_1

#print("something to display : ",np.where(norm_distances > seuil)[0])
anomalies_df = original_data_with_one_mat.iloc[anomalies_indices]


# In[363]:


anomalies_df


# In[364]:


df_mat = anomalies_df
#[anomalies_df['matricule'] == 388]
df_mat


# In[365]:


import matplotlib.pyplot as plt

df_mat_original = gp2_1
#[gp2_1['matricule'] == 388]

# Visualisation des données avec une distinction pour les anomalies
plt.plot(df_mat_original['date'], df_mat_original['quantite'], label='Données normales',color ='blue')
plt.scatter(df_mat['date'], df_mat['quantite'], color = 'r', label='Anomalies SOM',marker ='*')
plt.legend()
#rotate x-axis labels
for index, row in df_mat.iterrows():
     plt.annotate((str(row['diff_date'])), (row['date'], row['quantite']), textcoords="offset points", xytext=(10,7), ha='center', fontsize=10)

plt.xlabel('Date')
plt.ylabel('Quantité de carburant versée en L')
plt.title('Visualisation des données avec les anomalies détectées 932')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# ## --------------------------------------- 2 : AutoEncoders ------------------------------------------------

# In[226]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, TimeDistributed, RepeatVector,Dropout
from tensorflow.keras import Sequential


# ### Importation et prétraitement des données

# In[323]:


gp2_carburant = pd.read_csv('/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date.csv',delimiter =',')
data = gp2_carburant
df = data

df['date'] = pd.to_datetime(df['date'],format = '%Y/%m/%d')
# Calculer la différence de date entre chaque transaction et la suivante pour chaque matricule
#df['difference_de_date'] = df.groupby('matricule')['date'].diff().dt.total_seconds()
#df = df.dropna(subset=['difference_de_date'])

#df_clear = df[df['anomaly1'] == 1]
#df_clear.set_index('date',inplace=True)

#df_clear = df_clear[['quantite']]

#df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

df = df[df['matricule'] == 230]
df.head()


# In[324]:


# traitement de df  : le dataframe complet 
df.set_index('date',inplace=True)

df = df[['quantite']]
df.head()


# In[325]:


#%matplotlib inline

import seaborn as sns
plt.plot(df.index,df['quantite'])

#rotate x-axis labels
plt.xticks(rotation=45)


# In[289]:


#startdate = df['date'].min()
#enddate = df['date'].max()

#print('start date : ',startdate)
#print('End date : ',enddate)


# ### Création du modèle
# 

# In[290]:


from tensorflow.keras.metrics import Accuracy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def LSTMAE(nbr_entries, width,loss= "MSE") :
    
    valid_shape = (width,nbr_entries)
    model = keras.Sequential()
    
    #encoder
    model.add(LSTM(128, activation ='relu', input_shape =(width,nbr_entries),return_sequences = True))
    model.add(Dense(64,activation='relu'))
    #model.add(Dropout(rate=0.2))
    model.add(Dense(32,activation='relu'))

    #bottleneck
    model.add(Dense(units=10,activation='relu',name ='latent'))
    #model.add(BatchNormalization())
    
    #doceder
    model.add(Dense(32,activation = 'relu'))
    #model.add(Dropout(rate = 0.2))
    model.add(Dense(64,activation = 'relu'))
    model.add(LSTM(128, activation ='sigmoid', return_sequences = True))
    #la couche TimeDistributed crée un vecteur dont la longueur est égale au nombre de sorties de la couche précédente
    model.add(TimeDistributed(Dense(nbr_entries)))
    
    #model.add(LSTM(128, input_shape=valid_shape,return_sequences = True))
    #model.add(Dense(32,activation='relu'))
    #model.add(Dense(units=10,activation='relu',name ='latent'))
    #model.add(Dropout(rate=0.2))
    #model.add(RepeatVector(nbr_entries))
    #model.add(Dense(32,activation='relu'))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(Dropout(rate=0.2))
    #model.add(TimeDistributed(Dense(nbr_entries)))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.01)
    model.compile(optimizer = optimizer,loss =loss)
    
    return model


# In[291]:


WINDOW_SIZE = 10


# #### Formatage et preprocessing des données

# Scaling : pour améliorer les performances de modèle lors de l'entrainement

# In[292]:


def build_rolling_data(df_clear,window):
    
    #print(df.strides)
    rolling_data = np.lib.stride_tricks.as_strided(df_clear, (len(df_clear) - window + 1, window),(df_clear.strides[0],df_clear.strides[0]))
    
    #ajouter une dimension à la fin    
    rolling_data = tf.expand_dims(rolling_data,axis = -1)
    return rolling_data


# In[293]:


scaler = MinMaxScaler()
donnes = df.values
#print(donnes.shape())
#print(len(donnes))
normalized_data = scaler.fit_transform(donnes.reshape(-1,1))
#print(normalized_data)

rolling_normalized_data = build_rolling_data(normalized_data,WINDOW_SIZE)
#print(rolling_normalized_data.shape)


# In[294]:


#scaler = MinMaxScaler()
#donnes = df.values

# Créer une liste pour stocker les séquences normalisées
#normalized_sequences = []

# Créer un scaler pour chaque séquence
#scaler = MinMaxScaler()

# Normaliser chaque séquence individuellement
#for sequence in donnes:
    # Appliquer le MinMaxScaler à chaque séquence
#    normalized_sequence = scaler.fit_transform(sequence.reshape(-1,1))
#    normalized_sequences.append(normalized_sequence)

# Convertir la liste de séquences normalisées en un tableau numpy
#normalized_data = np.array(normalized_sequences)
#rolling_normalized_data = build_rolling_data(normalized_data,WINDOW_SIZE)


# ### Apprentissage du modèle

# In[295]:


model = LSTMAE(1, WINDOW_SIZE)


# In[296]:


import time

start = time.time()
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()


# In[297]:


history = model.fit(rolling_normalized_data,rolling_normalized_data, epochs = 50, batch_size = 256,verbose = 1)
print(f"model trained in {(time.time()-start) / 60} minutes")
print(history)


# ### Prediction

# In[298]:


import numpy as np


# In[299]:


#scaler = MinMaxScaler()
#donnes = df.values
#print(donnes.shape())
#print(len(donnes))
#normalized_data1 = scaler.fit_transform(donnes.reshape(-1,1))
#print(normalized_data)

#rolling_normalized_data1 = build_rolling_data(normalized_data1, WINDOW_SIZE)
#print(rolling_normalized_data.shape)


# In[300]:


predicted = model.predict(rolling_normalized_data)


# In[312]:


reconstruction_error = np.mean((rolling_normalized_data - predicted) ** 2, axis = 1)
len(reconstruction_error)


# In[313]:


decision = - reconstruction_error + np.mean(reconstruction_error) + 1 * np.std(reconstruction_error)


# In[314]:


#decision


# In[315]:


padding = [0]*(len(df) - len(decision)) 
padding


# In[316]:


#decision


# In[317]:


final_df = df
final_df['predict'] = list(decision.flatten()) + padding
final_df['predict'].sample(10)


# In[318]:


final_df['predict'] = final_df['predict'].apply(lambda x : -1 if x < 0 else 1)


# In[319]:


final_df


# In[320]:


#df = df.drop(['target'],axis =1)
#final_df = final_df.assign(IF_pred=data["anomaly1"].to_numpy())
#final_df.sample(5)


# In[321]:


final_df[final_df['predict'] != 1 ]


# #### Visualisation

# In[322]:


anomalies = final_df[final_df['predict'] == -1]
plt.plot(final_df['quantite'],label='Données originales', color='blue')
plt.scatter(anomalies.index,anomalies['quantite'],color='red', label='Anomalies détectées AE', marker='o')
plt.xlabel('Date')
plt.ylabel('Quantité de carburant versée en L')
plt.title('Visualisation des données avec les anomalies détectées - AE')
plt.legend()
plt.xticks(rotation=45)


# In[3995]:


from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(final_df['IF_pred'],final_df['predict'])
print(confusion_matrix)

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['False', 'True'])
#cm_display.plot()
#plt.show()


# ## -------------------------------AutoEncoder Simple  (sans LSTM) ---------------------------------

# In[368]:


import pandas as pd

df2 = gp2_carburant = pd.read_csv('/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date.csv',delimiter =',')

df2['date'] = pd.to_datetime(df2['date'],format = '%Y/%m/%d')
# traitement de df  : le dataframe complet 
df2.set_index('date',inplace=True)
df2 = df2[df2['matricule'] == 1624]

df2 = df2[['quantite']]
df2.head()


# In[399]:


from tensorflow.keras.metrics import Accuracy

def LSTMAE1(loss= "MAE") :
    
    model1 = Sequential()
    
    #encoder
    model1.add(Dense(64, activation ='relu'))
    model1.add(Dense(32,activation='relu'))
    #model1.add(Dropout(rate=0.2))
    model1.add(Dense(16,activation='relu'))

    #bottleneck
    model1.add(Dense(units=8,activation='relu'))
    model1.add(BatchNormalization())
    
    #doceder
    model1.add(Dense(16,activation = 'relu'))
    model1.add(Dropout(rate = 0.2))
    model1.add(Dense(32,activation = 'relu'))
    model1.add(Dense(64, activation ='sigmoid'))
    
    #model.add(TimeDistributed(Dense(nbr_entries)))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.1)
    model1.compile(optimizer = optimizer,loss = loss)
    
    return model1                                   


# In[400]:


#df2


# In[401]:


from sklearn.preprocessing import MinMaxScaler

#Normaiser les données
scaler1 = MinMaxScaler()
donnes_1 = df2.values
#print(donnes.shape())
#print(len(donnes))
normalized_data_1 = scaler1.fit_transform(donnes_1.reshape(-1,1))
#print(normalized_data)


# In[402]:


model1 = LSTMAE1()


# In[403]:


import time

start = time.time()
model1.compile(optimizer='adam',loss='mae')


# In[404]:


#model1.summary()


# In[405]:


history1 = model1.fit(normalized_data_1,normalized_data_1, epochs = 50, batch_size = 256,verbose = 1)
print(f"model trained in {(time.time()-start) / 60} minutes")
print(history1)


# #### Prediction

# In[406]:


#Normaiser les données

scaler = MinMaxScaler()
donnes_2 = df2.values

normalized_data_2 = scaler.fit_transform(donnes_2.reshape(-1,1))


# In[407]:


predicted_1 = model1.predict(normalized_data)


# In[408]:


import numpy as np
reconstruction_error1 = np.mean((normalized_data_1 - predicted_1), axis = 1)
len(reconstruction_error1)


# In[409]:


decision1 = - reconstruction_error1 + np.mean(reconstruction_error1) + 1 * np.std(reconstruction_error1)


# In[410]:


df1 = df2
#df1 = df1.drop(['predict'], axis = 1)


# In[411]:


#decision1


# In[412]:


padding = [0]*(len(df2) - len(decision1)) 
padding


# In[413]:


final_df_1 = df1
print(df1)
final_df_1['predict'] = list(decision1.flatten()) + padding
final_df_1['predict'].sample(5)


# In[414]:


final_df_1['predict'] = final_df_1['predict'].apply(lambda x : -1 if x < 0 else 1)


# In[415]:


final_df_1['predict'] 


# In[416]:


#plt.plot(final_df_1[['quantite','predict']])
#plt.xticks(rotation = 45) 
import matplotlib.pyplot as plt

anomalies = final_df_1[final_df_1['predict'] == -1]
plt.plot(final_df_1['quantite'],label='Données originales', color='blue')
plt.scatter(anomalies.index,anomalies['quantite'],color='red', label='Anomalies détectées AE 2', marker='o')

plt.xlabel('Date')
plt.ylabel('Quantité de carburant versée en L')
plt.title('Visualisation des données avec les anomalies détectées - AE')
plt.legend()
plt.xticks(rotation=45)


#---------------------------------------- 3 : Isolation Forest -------------------------------------------
#%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
from mplcursors import cursor
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

dataframe = pd.read_csv("/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date_2.csv",delimiter=',')
dataframe['date'] = pd.to_datetime(dataframe['date'], format='%Y/%m/%d')
dataframe
dataframe.head()
#dataframe['date'] = dataframe['date'].sort_values()

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = StandardScaler()
minmaxscaler = MinMaxScaler()

def train_isolation_forest_model(mat_dataset):
    
    string_model = "models/"+str(mat_dataset['matricule'].unique())+ ".pkl"

    data = mat_dataset[['quantite','type_carburant_enc','typePaiement_enc','diff_date','gouvernorat_enc']]
    #print(data['matricule'][0])
    np_scaled = minmaxscaler.fit_transform(data)

    #data = pd.DataFrame(np_scaled)

    # Isolation forest model
    outliers_fraction = 0.3
    random_state = np.random.RandomState(42)

    ifo_model = IsolationForest(n_estimators=100,
                                contamination='auto', 
                                max_features = 4, 
                                #max_samples = 0.5,
                                bootstrap=False, 
                                #random_state=1234,
                                warm_start = True,
                                #n_jobs=-1, 
                                verbose = 1,
                                random_state=0
                               )

    ifo_model.fit(data)
    mat_dataset['anomaly_if'] = ifo_model.predict(data)
    mat_dataset['anomaly_score'] = ifo_model.decision_function(data)
    joblib.dump(ifo_model, string_model)

    return mat_dataset,ifo_model


# In[ ]:


#récuperer la liste des matricules
matricules = dataframe['matricule'].unique()
len(matricules)


# ### MLFLOW Test For Isolation Forest Model Tracking

# In[37]:


import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment(experiment_id="0")


# In[38]:


data_per_mat = dataframe[dataframe['matricule'] == 230]
    
if_dataset,m = train_isolation_forest_model(data_per_mat)


# In[ ]:


#params = {
#    "solver": "lbfgs",
#    "max_iter": 1000,
#    "multi_class": "auto",
#    "random_state": 8888,
#}

# Start an MLflow run
#with mlflow.start_run():
#    # Log the hyperparameters
#    mlflow.log_params(params)

    # Set a tag that we can use to remind ourselves what this run was for
#    mlflow.set_tag("Training Info", "Basic LR model for iris data")


# In[ ]:


if_final = pd.DataFrame()
print(if_final)

#mlflow.set_experiment(experiment_id="0")
m = 0 
for mat in matricules :
    
    data_per_mat = dataframe[dataframe['matricule'] == mat]
    
    if_dataset,m = train_isolation_forest_model(data_per_mat)
    if_final = pd.concat([if_final, if_dataset], ignore_index = True)
    
#print(mat)
#print(m)    

#print(if_final.shape)


# In[ ]:


#enregistrer le dataset 
if_final.to_csv('if_dataset.csv', index=False)


# In[ ]:


my_saved_model = joblib.load('models/[230].pkl')

transactions = pd.DataFrame({
    #'matricule': [230, 230, 230,230,230],
    'quantite': [52.09, 56.65, 56.65,68.09,68.09,444,50.0,75.0,30,43.00,76.0,80.33,82.98,100.00,98.08,25.09,30,49.00,70.9,75.00],
    'type_carburant': [1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0],
    'typePaiement': [31,32,32,32,32,
                     32,32,32,32,32,30,32,32,32,32,32,32,32,32,32],
    #'fournisseur' : [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'diff_date': [2.0,2.0,2.0,2.0,1.0,2.0,1.0,2.0,1.0,2.0,2.0,2.0,2.0,1.0,2.0,0.0,2.0,4.0,2.0,3.0]  # Différence de date en jours
})


# In[ ]:


#transactions


# In[ ]:


np_scaled = minmaxscaler.fit_transform(transactions)
#np_scaled = pd.DataFrame(np_scaled)
transactions['anomaly_if'] = my_saved_model.predict(transactions)


# In[ ]:


transactions['real_labels'] = [-1,-1,1,1,1,-1,1,1,-1,
                              1,
                               -1,1,1,-1,-1,-1,-1,1,-1,1] 
transactions


# ### <font color= "green"> Evaluation des performances de Isolation Forest </font>

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prédictions de votre modèle
y_pred = transactions['anomaly_if'] 

# Calcul de l'accuracy
y_true = transactions['real_labels'] 
accuracy = accuracy_score(y_true, y_pred)

# Calcul de la précision
precision = precision_score(y_true, y_pred)

# Calcul du rappel
recall = recall_score(y_true, y_pred)

# Calcul du F1 Score
f1 = f1_score(y_true, y_pred)
print('Précision :',precision)
print('\n')
print('Recall :',recall)
print('\n')
print('Accuracy :',accuracy)


# ### <font color ='green'>Afficher un exemple de données pour une matricule spécifique </font>

# In[ ]:


res = if_final[(if_final['matricule'] == 230) & (if_final['anomaly_if'] == -1)]
res_1 = if_final[(if_final['matricule'] == 230) ]
print(len(res_1))

res.head()


# ### <font color = 'orange'> Visualisation Interactive de résultat d'une matricule</font>
# 

# In[ ]:


import plotly.io as pio
pio.renderers.default = "notebook_connected"


# In[ ]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Convertir les dates en chaine
#res_1['date'] = res_1['date'].astype(str)
anomalies = res_1[res_1['anomaly_if'] == -1]

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


# ###  -------------------------------------Test de Isolation Forest H2O Implémentation------------------------------------
# 

# In[1221]:


import h2o
from h2o.estimators import H2OIsolationForestEstimator
h2o.init()


# In[803]:


# Import the prostate dataset
h2o_df = h2o.import_file("/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/gp2_carburant_transformed_diff_date.csv")
h2o_df = h2o_df[h2o_df['matricule'] == 230]
# Split the data giving the training dataset 75% of the data
#train,test = h2o_df.split_frame(ratios=[0.75])

# Build an Isolation forest model
model = H2OIsolationForestEstimator(
                                    max_depth = 0,
                                    ntrees = 50,
                                    col_sample_rate_change_per_level =1.5,
                                    #sample_size = 1,
                                    #min_rows = 1,
                                    score_tree_interval = 1)
model.train(training_frame = h2o_df)

# Calculate score
score = model.predict(h2o_df)
h2o_df['prediction'] = score["predict"]

# Predict the leaf node assignment
ln_pred = model.predict_leaf_node_assignment(h2o_df, "Path")


# ### Données de test (manuellement générées)

# In[810]:


# Calculate score
hf = h2o.H2OFrame(transactions)

score_transactions = model.predict(hf)
hf['prediction'] = score_transactions["predict"]


# In[811]:


hf


# In[797]:


h2o_df


# In[804]:


ln_pred


# In[806]:


# Define anomaly threshold
anomaly_threshold = 0.45  # You can adjust this threshold as needed

# Identify anomalies based on anomaly threshold
anomalies = h2o_df[h2o_df['prediction'] > anomaly_threshold]

# View anomalies
print(anomalies)


# In[813]:


h2o_df = h2o_df.drop(['prediction'],axis = 1)


# In[50]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


# Convertir les dates en chaine
#res_1['date'] = res_1['date'].astype(str)
df = h2o_df.as_data_frame()
anomalies = anomalies.as_data_frame()
# Ajouter la trace principale pour les données
fig = px.line(df, x='date', y='quantite',title='Visualisation des données avec les anomalies détectées',
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


# In[ ]:


data_per_mat = if_final[if_final['matricule'] == 230]
print(data_per_mat.head())
data_per_mat_1= data_per_mat.drop(['date','anomaly_if','station'],axis =1)
#data_per_mat_1 = scaler.fit_transform(data_per_mat_1)
#data_per_mat['anomay_scores'] = my_saved_model.decision_function(data_per_mat_1)
#data_per_mat['anomay_scores']


# ###  Évaluation de modèle   Isolation Forest

# In[51]:


import matplotlib.pyplot as plt

# Tracer un histogramme des scores de détection d'anomalies

plt.hist(data_per_mat['anomaly_score'], bins=50, color='blue', alpha=0.5)
plt.xlabel('Score de détection d\'anomalies')
plt.ylabel('Fréquence')
plt.title('Distribution des scores de détection d\'anomalies')
plt.show()


# **1 - La métrique Silouhette**

# In[52]:


from sklearn.metrics import silhouette_score

# Calcul de la métrique silhouette
res_2 = res_1.drop(['date'],axis = 1)

silhouette_avg = silhouette_score(res_2, res_1['anomaly_if'])
print("Silhouette Score:", silhouette_avg)


# In[53]:


import numpy as np

# Calcul de la stabilité en utilisant les prédictions de l'Isolation Forest
def compute_stability(predictions, num_runs=10):
    n_samples = len(predictions)
    stability_scores = []
    for _ in range(num_runs):
        random_predictions = np.random.choice(predictions, n_samples, replace=False)
        stability_score = np.mean(predictions == random_predictions)
        stability_scores.append(stability_score)
    return np.mean(stability_scores)


# In[54]:


# Supposons que anomaly_predictions sont les prédictions de votre modèle Isolation Forest

stability_score = compute_stability(res_1['anomaly_if'])
print("Stability Score:", stability_score)


# In[55]:


import seaborn as sns
import pandas as pd

# Supposons que vous avez déjà votre DataFrame 'data' contenant vos données

# Tracez un pairplot pour visualiser les relations entre les caractéristiques
#res_2 = res_1.drop(['matricule','anomaly_if'],axis = 1)
#sns.pairplot(res_2)
#plt.show()


# ### <font color ='green'>Résultat final obtenu grâce à Isolation Forest</font>

# In[56]:


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
