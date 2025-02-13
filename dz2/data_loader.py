import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import logging


logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s")

class DataLoader:
    logger = logging.getLogger(__name__)
    logger.info('Started modul DataLoader')

    def load_csv(self, file_path):
        try:
            data = pd.read_csv(file_path)
            self.logger.info("Данные успешно загружены")
            return data
        except Exception as e:
            self.logger.exception(f"Ошибка при загрузке CSV: {e}")
            return None

    def load_json(self, file_path):
        try:
            data = pd.read_json(file_path)
            self.logger.info("Данные успешно загружены")
            return data
        except Exception as e:
            self.logger.exception(f"Ошибка при загрузке JSON: {e}")
            return None

    def load_api(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            self.logger.info("Данные успешно загружены")
            return pd.json_normalize(data)
        except Exception as e:
            self.logger.exception(f"Ошибка при загрузке из API: {e}")
            return None

