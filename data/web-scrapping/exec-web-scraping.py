
import pandas as pd
from Station_web_scraping import Web_Scrapping

gp2_carburant = pd.read_csv('data/input-data/gp2_carburant.csv',delimiter=',',encoding='utf-8')
gp2_carburant = gp2_carburant[['quantite','date','fournisseur','note','station']]
x  = Web_Scrapping().web_scrapping_station(gp2_carburant.head())
print(x)