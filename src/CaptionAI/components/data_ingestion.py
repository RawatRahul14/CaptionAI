import time
import os
import requests
from kagglehub import dataset_download
from CaptionAI import logger
from CaptionAI.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if os.path.exists(self.config.local_data_file):
            logger.info("Dataset already exists.")
            return

        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Attempt {attempt + 1} to download the dataset...")
                
                data_path = dataset_download(self.config.dataset_link)
                logger.info("Dataset has been downloaded successfully.")

                os.rename(src = data_path,
                          dst = self.config.local_data_file)
                
                logger.info(f"Dataset has been moved to: {self.config.local_data_file}")
                return  
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout occurred during download attempt {attempt + 1}. Retrying...")
            except Exception as e:
                logger.error(f"Error occurred during download: {e}")
                if attempt == retries - 1:
                    raise  
            time.sleep(2 ** attempt)  