import logging
import requests
import tempfile
import dotenv
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
dotenv.load_dotenv(".env_d3")

class StellaD3():
    def __init__(self, local_save_dir:str="outputs", local_save:bool=True, show_debug_log:bool=False):
        try:
            if show_debug_log: logging.basicConfig(level=logging.DEBUG)
            
            self.local_save = local_save
                
            if self.local_save:
                self.local_save_dir = local_save_dir
                os.makedirs(self.local_save_dir, exist_ok=True)
            
            else: self.local_save_dir = None
                
            options = Options()
            options.add_argument("--headless=new")
            options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            self.driver = webdriver.Chrome(options=options)      
            self.driver.get(os.getenv("CONNECTOR_URL"))
            
            email_input = WebDriverWait(self.driver, 2).until(EC.visibility_of_element_located((By.ID, "i0116")))
            email_input.send_keys(os.getenv("CONNECTOR_ID"))
            
            next_button = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((By.ID, "idSIButton9")))
            next_button.click()
            
            password_input = WebDriverWait(self.driver, 2).until(EC.visibility_of_element_located((By.ID, "i0118")))
            password_input.send_keys(os.getenv("CONNECTOR_KEY"))
            
            next_button = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((By.ID, "idSIButton9")))
            next_button.click()
            
            accept_button = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((By.ID, "acceptButton")))
            accept_button.click()
            
            print('D3 initialized!')
    
        except Exception as e:
            print(f'Error: {e}')
            quit()
    
    def image_generator_d3(self, prompt):
        self.driver.get(os.getenv("CREATOR_URL"))
        self.driver.refresh()
        
        self.driver.find_element(By.ID, "sb_form_q").send_keys(prompt)
        self.driver.find_element(By.ID, "create_btn_c").click()

        output_files = []
    
        if self.local_save:
            temp_dir = str(self.local_save_dir)
            temp_dir = os.path.join(temp_dir, datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(temp_dir, exist_ok=True)
        
        else: temp_dir = None
        
        try:
            WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "gil_err_tc")))
            raise Exception('GIL_ERR_TC (Prompt Blocked)')
        
        except TimeoutException:
            pass

        while True:
            print("GI_REFRESH (Refresh Initiated)")
            self.driver.refresh()

            try:
                WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "img_cont")))
                divs = self.driver.find_elements(By.CLASS_NAME, "img_cont")
                urls = [div.find_element(By.TAG_NAME, "img").get_attribute("src").split("?")[0] for div in divs]
                print('IMG_CONT (Complete!)')
                
                for url in urls:
                    response = requests.get(url)

                    with tempfile.NamedTemporaryFile(delete=False, prefix=f"{self.__class__.__name__}_", suffix=".png", dir=temp_dir) as output:
                        output.write(response.content)
                        output_files.append(output.name)

                return output_files
                
            
            except TimeoutException:
                try:
                    img = self.driver.find_element(By.CLASS_NAME, "gir_mmimg")
                    src = img.get_attribute("src").split("?")[0]
                    print('GIR_MMIMG (Complete!)')
                    response = requests.get(src)
                    
                    with tempfile.NamedTemporaryFile(delete=False, prefix=f"{self.__class__.__name__}_", suffix=".png", dir=temp_dir) as output:
                        output.write(response.content)
                        output_files.append(output.name)

                    return output_files
                
                except NoSuchElementException:
                    raise Exception('GIL_NOT_FOUND (Element Not Found)')

if __name__ == "__main__":
    print("Import StellaD3 in another file to start using it!")