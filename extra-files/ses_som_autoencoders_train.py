

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

# In[124]:


df3 = pd.read_csv('/Users/abdelwahed/projet-pfe-detecteur-fraude-carburant/gp2_carburant_transformed.csv',delimiter =";"
                
                 )

df3['Date'] = pd.to_datetime(df3['date'])
# Définir la colonne 'Date' comme index
df3.set_index('date', inplace=True)
df = df3[df3['matricule'] == 230]
df = df[['quantite']]


# In[125]:


len(df)


# In[126]:


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

# In[127]:


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

# In[128]:


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

# In[130]:


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


# In[132]:


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


# In[133]:


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


# In[134]:


anomalies


# In[135]:


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
# * Nous remarquons que les résulats de prédiction de SES n'ont pas performants.
# 
# * En effet, pour de bonnes prédictions de série temorolles ,il faut utiliser de données sans anomalies, mais ce n'est pas le ca pour nous.

# # Apprentissage non supervisé pour la détection d'anomalies
# 

# ## ------------------------------ 1 : SOM : Self - Organized Maps -------------------------------------

# In[464]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import bone, pcolor, colorbar, plot, show
import pylab as pl


# In[465]:


#pip install minisom


# In[466]:


gp2_carburant = pd.read_csv('/Users/abdelwahed/projet-pfe-detecteur-fraude-carburant/gp2_carburant_transformed.csv',delimiter =';')

gp2_carburant['date'] = pd.to_datetime(gp2_carburant['date'],format='%d/%m/%Y %H:%M')
gp2_carburant.head()


# In[467]:


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


# In[468]:


data = gp2_carburant
#d = data
#d['Year'] = d['date'].dt.year
#d['Month'] = d['date'].dt.month
#d['Day'] = d['date'].dt.day
#d['DayOfWeek'] = d['date'].dt.dayofweek
#d['DayOfYear'] = d['date'].dt.dayofyear
#d['WeekOfYear'] = d['date'].dt.isocalendar().week
data.head()


# In[463]:


#d[d['matricule'] == 230].head()


# In[469]:



# Trier les données par matricule puis par date (au cas où ce n'est pas déjà fait)
gp2 = gp2_carburant
gp2 = gp2.sort_values(by=['matricule', 'date'])

# Calculer la différence de date entre chaque transaction et la suivante pour chaque matricule
#gp2['difference_de_date'] = gp2.groupby('matricule')['date'].diff().dt.total_seconds()
#gp2 = gp2.dropna(subset=['difference_de_date'])

gp2.head()


# #### Trouver les différences en jours entre les dates consécutives

# In[358]:


# Trouver la différence la plus fréquente
all_matricules = gp2['matricule']
all_matricules = all_matricules.unique()
new_dataframe = pd.DataFrame()


# In[359]:


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

    #supprimer la premiere observation
    #sub_data_frame = sub_data_frame.iloc[1:]   
    
    #sub_data_frame['diff_date'] = diffs
    #sub_data_frame = sub_data_frame.dropna(subset=['diff_date']).dropna()

    new_dataframe = pd.concat([new_dataframe, sub_data_frame], ignore_index = True)
    


# In[470]:


new_dataframe['diff_date'].isnull().sum()


# In[471]:


new_dataframe['diff_date']


# In[472]:


new_dataframe.to_csv('gp2_carburant_transformed_diff_date.csv', index=False)


# In[473]:


gp2 = new_dataframe

gp2_1 = gp2
#gp2_1 = gp2.drop(['Year','Month','DayOfWeek','DayOfYear','WeekOfYear','Day','station'],axis =1)
gp2 = gp2[gp2['matricule'] == 932]
gp2_1 = gp2_1[gp2_1['matricule'] == 932]

#gp2 = gp2.drop(['date','Year','Month','DayOfWeek','DayOfYear','WeekOfYear','Day','station'],axis =1)
#gp2 = gp2[['matricule','quantite']]
gp2.head()


# In[474]:


gp2 = gp2.drop(['date','station'], axis =1)


# ### Feature Scaling

# In[475]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
gp2_scaled = sc.fit_transform(gp2)


# In[476]:


from sklearn.preprocessing import StandardScaler
 
object= StandardScaler()
  
# standardization 
gp2_scaled = object.fit_transform(gp2) 
#print(scale)


# ### Training the SOM
# 

# In[477]:


map_size = 5 * math.sqrt(len(gp2_scaled))
map_height = map_width = math.ceil(math.sqrt(map_size))


# In[478]:


print(f'(map_height, map_width) = ({map_height}, {map_width})')
print(f'Number of features: {gp2_scaled.shape[1]}')


# In[479]:


gp2_scaled.shape[1]


# In[480]:


from minisom import MiniSom

som = MiniSom(x=map_width, y=map_height, input_len = gp2_scaled.shape[1], sigma = 1.4, learning_rate =0.001
              ,neighborhood_function='gaussian')

som.random_weights_init(gp2_scaled)
som.train_batch(data = gp2_scaled, num_iteration = 1000, verbose = True)


# In[481]:


import numpy as np

weights = som.get_weights()

# Enregistrement des poids dans un fichier
np.save("som_weights.npy", weights)


# In[482]:



# Charger les poids à partir du fichier
#weights = np.load("som_weights.npy")

# Créer un nouveau modèle SOM avec les poids chargés
#new_model = MiniSom(x=map_width-1, y=map_height-1, input_len=gp2_scaled.shape[1], sigma = 1.4,learning_rate =0.1) 
#new_som.weights = weights.reshape((gp2_scaled, new_model._weights.shape[1], gp2_scaled.shape[1]))


# ### Comprendre les résultats 
# 
# Le nœud gagnant est celui qui est le plus proche de l’entité d’entrée Xi.
# 
# Nous pouvons obtenir ses coordonnées (x, y) sur la carte en utilisant la méthode winner().

# In[483]:


som.winner(gp2_scaled[0])


# La carte de distance est un tableau 2D (17x17) où chaque élément représente la distance moyenne entre un neurone et ses voisins.

# In[484]:


print('-------------\nDistance Map\n------------')
print(f'Shape: {som.distance_map().shape}')
print(f'First Line: {som.distance_map().T[0]}')


# ### Les fréquences : array 2D qui contient chaque neuronne combien de fois a gagné comme BMU
# Exemple : neuronne localisé à (2, 3) à gagné 3 fois comme BMU

# In[495]:


frequencies = som.activation_response(gp2_scaled)
print(f'Fréquences:\n {np.array(frequencies.T, np.uint)}')


# #### Visualiser les resultats

# In[486]:


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

# In[494]:


plt.figure(figsize=(7, 7))
fréquences = som.activation_response (gp2_scaled)
plt.pcolor(frequencies.T, cmap='Blues')
plt.colorbar()
plt.show()


# ### Identifier les anomalies

# In[379]:


# Calculer la distance à BMU pour chaque donnée
distances_to_bmu = np.array([som.winner(d) for d in gp2_scaled])

# Calculer la norme de ces distances
norm_distances = np.linalg.norm(distances_to_bmu, axis=1)
print(type(norm_distances))


# In[488]:


# Définir un seuil
seuil = np.mean(norm_distances) + 1 * np.std(norm_distances)
print('seuil : ',seuil)

# Identifier les indices des anomalies
anomalies_indices = np.where(norm_distances > seuil)

#print(anomalies_indices)
original_data_with_one_mat = gp2_1

#print("something to display : ",np.where(norm_distances > seuil)[0])
anomalies_df = original_data_with_one_mat.iloc[anomalies_indices]


# In[489]:


anomalies_df


# In[490]:


df_mat = anomalies_df
#[anomalies_df['matricule'] == 388]
df_mat


# In[491]:


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

# In[654]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, TimeDistributed, RepeatVector,Dropout
from tensorflow.keras import Sequential


# ### Importation et prétraitement des données

# In[695]:


gp2_carburant = pd.read_csv('/Users/abdelwahed/projet-pfe-detecteur-fraude-carburant/gp2_carburant_transformed_diff_date.csv',delimiter =',')
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


# In[696]:


# traitement de df  : le dataframe complet 
df.set_index('date',inplace=True)

df = df[['quantite']]
df.head()


# In[697]:


#%matplotlib inline

import seaborn as sns
plt.plot(df.index,df['quantite'])

#rotate x-axis labels
plt.xticks(rotation=45)


# In[698]:


#startdate = df['date'].min()
#enddate = df['date'].max()

#print('start date : ',startdate)
#print('End date : ',enddate)


# ### Création du modèle
# 

# In[699]:


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
    model.add(Dropout(rate=0.2))
    model.add(Dense(32,activation='relu'))

    #bottleneck
    model.add(Dense(units=10,activation='relu',name ='latent'))
    #model.add(BatchNormalization())
    
    #doceder
    model.add(Dense(32,activation = 'relu'))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(64,activation = 'relu'))
    model.add(LSTM(128, activation ='sigmoid', return_sequences = True))
    #la couche TimeDistributed crée un vecteur dont la longueur est égale au nombre de sorties de la couche précédente
    model.add(TimeDistributed(Dense(nbr_entries)))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.01)
    model.compile(optimizer = optimizer,loss =loss)
    
    return model


# In[700]:
WINDOW_SIZE = 5
# #### Formatage et preprocessing des données

# Scaling : pour améliorer les performances de modèle lors de l'entrainement

# In[701]:

def build_rolling_data(df_clear,window):
    
    #print(df.strides)
    rolling_data = np.lib.stride_tricks.as_strided(df_clear, (len(df_clear) - window + 1, window),(df_clear.strides[0],df_clear.strides[0]))
    
    #ajouter une dimension à la fin    
    rolling_data = tf.expand_dims(rolling_data,axis = -1)
    return rolling_data


# In[702]:

scaler = MinMaxScaler()
donnes = df.values
#print(donnes.shape())
#print(len(donnes))
normalized_data = scaler.fit_transform(donnes.reshape(-1,1))
#print(normalized_data)

rolling_normalized_data = build_rolling_data(normalized_data,WINDOW_SIZE)
#print(rolling_normalized_data.shape)

# ### Apprentissage du modèle
# In[704]:
model = LSTMAE(1, WINDOW_SIZE)

# In[705]:
import time
start = time.time()
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

# In[706]:

history = model.fit(rolling_normalized_data,rolling_normalized_data, epochs = 20, batch_size = 512,verbose = 1)
print(f"model trained in {(time.time()-start) / 60} minutes")
print(history)

# ### Prediction

# In[707]:
import numpy as np
# In[708]:
# In[709]:
predicted = model.predict(rolling_normalized_data)

# In[721]:

reconstruction_error = np.mean((rolling_normalized_data - predicted) ** 1, axis = 1)
len(reconstruction_error)

# In[722]:
decision = - reconstruction_error + np.mean(reconstruction_error) + 1 * np.std(reconstruction_error)

# In[723]:

#decision

# In[724]:
padding = [0]*(len(df) - len(decision)) 
padding
# In[725]:

#decision
# In[726]:
final_df = df
final_df['predict'] = list(decision.flatten()) + padding
final_df['predict'].sample(10)

# In[727]:

final_df['predict'] = final_df['predict'].apply(lambda x : -1 if x < 0 else 1)
# In[728]:
final_df

# In[729]:
#df = df.drop(['target'],axis =1)
#final_df = final_df.assign(IF_pred=data["anomaly1"].to_numpy())
#final_df.sample(5)
# In[730]:
final_df[final_df['predict'] != 1 ]

# #### Visualisation

# In[731]:
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

# In[574]:
import pandas as pd

df2 = gp2_carburant = pd.read_csv('/Users/abdelwahed/projet-pfe-detecteur-fraude-carburant/gp2_carburant_transformed_diff_date.csv',delimiter =',')

df2['date'] = pd.to_datetime(df2['date'],format = '%Y/%m/%d')
# traitement de df  : le dataframe complet 
df2.set_index('date',inplace=True)
df2 = df2[df2['matricule'] == 230]

df2 = df2[['quantite']]
df2.head()


# In[575]:


from tensorflow.keras.metrics import Accuracy

def LSTMAE1(loss= "MAE") :
    
    model1 = Sequential()
    
    #encoder
    model1.add(Dense(64, activation ='relu'))
    model1.add(Dense(32,activation='relu'))
    model1.add(Dropout(rate=0.2))
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
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.02)
    model1.compile(optimizer = optimizer,loss = loss)
    
    return model1                                   


# In[576]:


#df2


# In[577]:


from sklearn.preprocessing import MinMaxScaler

#Normaiser les données
scaler1 = MinMaxScaler()
donnes_1 = df2.values
#print(donnes.shape())
#print(len(donnes))
normalized_data_1 = scaler1.fit_transform(donnes_1.reshape(-1,1))
#print(normalized_data)


# In[578]:


model1 = LSTMAE1()


# In[579]:


import time

start = time.time()
model1.compile(optimizer='adam',loss='mae')


# In[580]:


#model1.summary()


# In[581]:


history1 = model1.fit(normalized_data_1,normalized_data_1, epochs = 50, batch_size = 256,verbose = 1)
print(f"model trained in {(time.time()-start) / 60} minutes")
print(history1)


# #### Prediction

# In[582]:


#Normaiser les données

scaler = MinMaxScaler()
donnes_2 = df2.values

normalized_data_2 = scaler.fit_transform(donnes_2.reshape(-1,1))


# In[644]:


predicted_1 = model1.predict(normalized_data_1)


# In[645]:


import numpy as np
reconstruction_error1 = np.mean((normalized_data_1 - predicted_1), axis = 1)
len(reconstruction_error1)


# In[646]:


decision1 = - reconstruction_error1 + np.mean(reconstruction_error1) + 0.3 * np.std(reconstruction_error1)


# In[647]:


df1 = df2
#df1 = df1.drop(['predict'], axis = 1)


# In[648]:


#decision1


# In[649]:


padding = [0]*(len(df2) - len(decision1)) 
padding


# In[650]:


final_df_1 = df1
print(df1)
final_df_1['predict'] = list(decision1.flatten()) + padding
final_df_1['predict'].sample(5)


# In[651]:


final_df_1['predict'] = final_df_1['predict'].apply(lambda x : -1 if x < 0 else 1)


# In[652]:


final_df_1['predict'] 


# In[653]:


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