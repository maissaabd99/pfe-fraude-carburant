import pandas as pd
from geopy.geocoders import Nominatim

class getGov ():

  def get_address(locality_name):
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(locality_name,language='fr', country_codes=['TN'])
    if location:
        return location.address
    else:
        return "Adresse non trouvée"
    
    
  def get_gouvernorat(self, gp2_carburant) :
    gp2_carburant['gouvernorat'] = ""
    i = 0
    zones = gp2_carburant['zone'].unique()
    dict_data = {'zone' :[],'gouvernorat' :[]}
    
    for zone in zones :
        dict_data['zone'].append(zone)
        address = self.get_address(zone)
        if address != "Adresse non trouvée" :
            parts = address.split("Gouvernorat")
            # Trouvez la partie de la chaîne après "Gouvernorat"
            part_after_gouvernorat = address.split("Gouvernorat")[1]
            print("part after gov : ",part_after_gouvernorat)
            # Trouvez la partie jusqu'à la première virgule après "Gouvernorat"
            words_after_gouvernorat = part_after_gouvernorat.split(',', 1)[0].strip()
            gp2_carburant['gouvernorat'][i] = words_after_gouvernorat
            print(gp2_carburant['gouvernorat'][i])
            dict_data['gouvernorat'].append(words_after_gouvernorat)
            print(f"Adresse complète de : {zone} est : ", words_after_gouvernorat)
            i+= 1 
            print("le compteur i est :",i)
        else :
            dict_data['gouvernorat'].append("not found")
            print("something went wrong !!")
            print("le compteur i est :",i)
            i +=1 
    # -------------------------------------part II---------------------------------
    final_dictionnaire= {}
    for cle, valeur in zip(dict_data['zone'],dict_data['gouvernorat']):
        final_dictionnaire[cle] = valeur
    print(dict_data)
    
    def mapper(zone):
        #print(dict_data.get(zone))
        return final_dictionnaire.get(zone)
     
    gp2_carburant['gouvernorat'] = gp2_carburant['zone'].map(mapper)
    #gp2_carburant.to_csv('gp2_carburant_transformed_diff_date_2024.csv', index=False)
    return gp2_carburant
    
