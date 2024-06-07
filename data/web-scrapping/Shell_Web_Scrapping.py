from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
# Part I : get all unique stations
#gp2_carburant = pd.read_csv('gp2_carburant_transformed_diff_date.csv',delimiter=',',encoding='utf-8')


class Web_scrapping_Station:
    final_dictionnaire = {}
    def mapper(self, localite):
        #localite = localite.lower()
        return self.final_dictionnaire.get(localite, 'Gouvernorat inconnu')[0]  # Par défaut, 'Gouvern||at inconnu' si la localité n'est pas trouvée
    
    def mapper2(self, localite):
        #localite = localite.lower()
        values = self.final_dictionnaire.get(localite)
        #print(values)
        if values and len(values) >= 3:
            return values[1], values[2]  # Retourne la latitude et la longitude
        else:
            return None, None  # Valeurs par défaut si localité non trouvée
    
    def web_scrapping_station (self, gp2_carburant) :
         gp2_carburant['gouvernorat'] = ""
         gp2_carburant['lat'] = ""
         gp2_carburant['lon'] = ""  
         # gp2_carburant.sample(5)
         # récuper la liste des stations ditinctes
         my_dict = {'station':[],'zone' : []}
         all_stations_carburant = gp2_carburant['station'].unique()
         #all_stations_carburant = [nom.lower() for nom in all_stations_carburant]
         print("Les noms des stations distincts disponibles : ",len(all_stations_carburant))
         search_results = None
         chrome_options = webdriver.ChromeOptions()
         chrome_options.add_argument('headless')
         chrome_options.headless = True 
         driver = webdriver.Chrome(chrome_options)
         driver.minimize_window()
         driver.set_window_position(100, 100)
         # Navigate to the Google search page
         # Find the search input element and send your search query
         i = 0
         for station in all_stations_carburant :  
            i+=1
            driver.get("https://www.google.com/")
            search_input = driver.find_element(By.NAME, "q")
            search_input.send_keys(station + " shell")
            search_input.send_keys(Keys.RETURN)
            wait = WebDriverWait(driver, 5)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "MjjYud")))
            
            # search for Shell site resultats
            search_results = driver.find_elements(By.CLASS_NAME, "MjjYud")
            print("all results found in page 1 : ",len(search_results))
            #iterate through search_results
            for res in search_results :
                res2 = (res.find_element(By.XPATH, "//span[contains(@class,'VuuXrf')]").text)
                print("res is:::::::::::::: ", res2)
                if res2 == "Shell Global":
                    driver2 = webdriver.Chrome(chrome_options)
                    driver2.minimize_window()
                    driver2.set_window_position(100, 100)
                    a = res.find_element(By.XPATH, "//a[contains(@jsname,'UWckNb')]")
                    href_value = a.get_attribute("href")
                
                    # Open a new tab and navigate to the href value of each product
                    driver2.execute_script("window.open('" + href_value + "', '_blank');")
                
                    # Switch to the new tab of each product
                    driver2.switch_to.window(driver2.window_handles[1])
                
                    # Wait for the images to be present in the DOM
                    wait = WebDriverWait(driver2, 5)
                    wait.until(EC.presence_of_all_elements_located((By.XPATH, "*")))
                
                    address = driver2.find_element(By.CLASS_NAME, "station-page-details__value").text
                    latlong = driver2.find_element(By.XPATH, "//div[contains(@aria-labelledby,'details-lat_lng')]").text
                    print("lat et lon :", latlong)
                    # Trouvez la partie de la chaîne après "Gouvern||at"
                    #gov = address.split(", ")[2]
                    parts = address.split(',')
                    lat,long = latlong.split(',')
                    print(lat,long)
                    # Récupérer le dernier élément de la liste et supprimer les espaces au début et à la fin
                    last_item = parts[-2].strip()
                    print("la zone coupé est : ",last_item)
                    my_dict['station'].append(station)
                    if last_item != '' or last_item !=  None :
                        
                        my_dict['zone'].append([last_item,lat, long])
                        print("on est dans la station numéro :",i)
                        # print('adresse est :',address)
                        print("resultat final de ",station, "est :", last_item)
                    else :
                         my_dict['zone'].append("None")
                         
                    break
                else:
                    print("nope!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
               
         print('le resulats de recherche sont :')
         
         
         #---------------------------------Part III : Structure the final result---------------------------
         
         
         # Utilisation d'une boucle for pour peupler le dictionnaire
         for cle, valeur in zip(my_dict['station'],my_dict['zone']):
             self.final_dictionnaire[cle] = valeur
         
         print("final dict after transformation :",self.final_dictionnaire)
         
         # Affichage du dictionnaire
         print(self.final_dictionnaire)
         gp2_carburant.head()
         
         # Appliquer la fonction de mapping à la colonne 'localite' pour obtenir la colonne 'gouvern||at'
         gp2_carburant['zone'] = gp2_carburant['station'].map(self.mapper)
         gp2_carburant[['lat', 'lon']] = gp2_carburant['station'].apply(lambda x: pd.Series(self.mapper2(x)))
         #print(gp2_carburant[['lat', 'lon']])
         gp2_carburant.to_csv('gp2_carburant_transformed_diff_date_2.csv', index=False)
         return gp2_carburant
     
     