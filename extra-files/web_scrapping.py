from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys


try:
  search_results = None
  driver = webdriver.Chrome()
  driver.get("https://www.shell.tn/conducteurs/localisateur-de-station-shell.html")
  #Wait for the images to be present in the DOM
  wait = WebDriverWait(driver, 5)
  #wait.until(EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'app__sidebar app__sidebar--has-content')]")))
            
  #results_page1 = driver.page_source
  print("Page title:", driver.title)
  # Get search results
  search_input =  driver.find_elements(By.CLASS_NAME,"search__input")
  #search_input.send_keys("nabeul mc 27-hdidane")
  #search_input.send_keys(Keys.RETURN)
  wait = WebDriverWait(driver, 5)
  #wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "*")))
  #search_results1 =  driver.find_elements(By.CLASS_NAME,"info-window__title")

  print("------------------------------Résultats------------------------------------------")
  print(search_input)
  #print(search_results1)


except TimeoutException as e:
    print("Timeout occurred:", e)
driver.quit()
