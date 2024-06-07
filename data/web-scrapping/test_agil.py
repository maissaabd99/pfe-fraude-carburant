from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import requests

class Web_scrapping_Station:
    final_dictionnaire = {}

    def mapper(self, localite):
        return self.final_dictionnaire.get(localite, 'Gouvernorat inconnu')[0]

    def mapper2(self, localite):
        values = self.final_dictionnaire.get(localite)
        if values and len(values) >= 3:
            return values[1], values[2]
        else:
            return None, None

    def get_coordinates_nominatim(self, address):
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "countrycodes": "TN"
        }
        headers = {
            "User-Agent": "YourAppName/1.0 (maissa.abdelwahed99@gmail.com)"
        }
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                location = data[0]
                return location["lat"], location["lon"]
            else:
                print("No results found for address:", address)
                return None, None
        elif response.status_code == 403:
            print("Access forbidden: You may be sending too many requests. Try again later.")
            return None, None
        else:
            print("Error:", response.status_code)
            return None, None

    def web_scrapping_station(self, gp2_carburant):
        gp2_carburant['gouvernorat'] = ""
        gp2_carburant['lat'] = ""
        gp2_carburant['lon'] = ""
        
        my_dict = {'station': [], 'zone': []}
        all_stations_carburant = gp2_carburant['station'].unique()
        print("Les noms des stations distincts disponibles : ", len(all_stations_carburant))
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('headless')
        chrome_options.headless = True
        driver = webdriver.Chrome(options=chrome_options)
        driver.minimize_window()
        driver.set_window_position(100, 100)
        
        i = 0
        for station in all_stations_carburant:
            i += 1
            driver.get("https://www.google.com/")
            search_input = driver.find_element(By.NAME, "q")
            if "AGIL" in station.upper():
                search_input.send_keys(station + " AGIL")
            else:
                search_input.send_keys(station + " SHELL")
            search_input.send_keys(Keys.RETURN)
            wait = WebDriverWait(driver, 5)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "MjjYud")))
            
            search_results = driver.find_elements(By.CLASS_NAME, "MjjYud")
            print("all results found in page 1 : ", len(search_results))
            
            for res in search_results:
                if "AGIL" in station.upper():
                    driver2 = webdriver.Chrome(options=chrome_options)
                    driver2.minimize_window()
                    driver2.set_window_position(100, 100)
                    a = res.find_element(By.XPATH, "//a[contains(@jsname,'UWckNb')]")
                    href_value = a.get_attribute("href")
                    
                    driver2.execute_script("window.open('" + href_value + "', '_blank');")
                    driver2.switch_to.window(driver2.window_handles[1])
                    
                    wait = WebDriverWait(driver2, 5)
                    wait.until(EC.presence_of_all_elements_located((By.XPATH, "*")))
                    
                    address = driver2.find_element(By.CLASS_NAME, "station-page-details__value").text
                    lat, long = self.get_coordinates_nominatim(address)
                    parts = address.split(',')
                    last_item = parts[-2].strip()
                    
                    my_dict['station'].append(station)
                    my_dict['zone'].append([last_item, lat, long])
                    driver2.quit()
                    break
                else:
                    res2 = res.find_element(By.XPATH, "//span[contains(@class,'VuuXrf')]").text
                    if res2 == "Shell Global":
                        driver2 = webdriver.Chrome(options=chrome_options)
                        driver2.minimize_window()
                        driver2.set_window_position(100, 100)
                        a = res.find_element(By.XPATH, "//a[contains(@jsname,'UWckNb')]")
                        href_value = a.get_attribute("href")
                        
                        driver2.execute_script("window.open('" + href_value + "', '_blank');")
                        driver2.switch_to.window(driver2.window_handles[1])
                        
                        wait = WebDriverWait(driver2, 5)
                        wait.until(EC.presence_of_all_elements_located((By.XPATH, "*")))
                        
                        address = driver2.find_element(By.CLASS_NAME, "station-page-details__value").text
                        latlong = driver2.find_element(By.XPATH, "//div[contains(@aria-labelledby,'details-lat_lng')]").text
                        lat, long = latlong.split(',')
                        parts = address.split(',')
                        last_item = parts[-2].strip()
                        
                        my_dict['station'].append(station)
                        my_dict['zone'].append([last_item, lat, long])
                        driver2.quit()
                        break
                print("nope!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
               
        print('le resulats de recherche sont :')
         
        for cle, valeur in zip(my_dict['station'], my_dict['zone']):
            self.final_dictionnaire[cle] = valeur
         
        print("final dict after transformation :", self.final_dictionnaire)
         
        gp2_carburant['zone'] = gp2_carburant['station'].map(self.mapper)
        gp2_carburant[['lat', 'lon']] = gp2_carburant['station'].apply(lambda x: pd.Series(self.mapper2(x)))
        gp2_carburant.to_csv('gp2_carburant_transformed_diff_date_2.csv', index=False)
        return gp2_carburant

gp2_carburant = pd.read_csv('/Users/abdelwahed/Mise-en-place-dun-detecteur-de-fraude-de-carburant/data/input-data/gp2_carburant.csv',delimiter=';',encoding='utf-8')
gp2_carburant = gp2_carburant[['quantite','date','fournisseur','note','station']]
d = Web_scrapping_Station()
x  = d.web_scrapping_station(gp2_carburant.head())
print(x)
