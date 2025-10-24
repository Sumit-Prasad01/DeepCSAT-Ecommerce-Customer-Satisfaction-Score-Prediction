import numpy as np
import pandas as pd
import os 
import sys
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import * 
from utils.common_functions import read_yaml
from sklearn.model_selection import train_test_split


logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, config):
        self.config = config['data_ingestion']
        self.train_test_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok = True)
        logger.info(f" Starting Data Ingestion")

    
    def split_data(self):
        try:
            logger.info("Starting splitting process")
            data = pd.read_csv(RAW_FILE_PATH)

            tarin_data, test_data = train_test_split(data, test_size = 1 - self.train_test_ratio, random_state = 42)

            tarin_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error While splitting data")
            raise CustomException("Failed to split data into train and test sets", e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion Process")

            self.split_data()

            logger.info("Data Ingestion Completed Successfully")

        except CustomException as ce:
            logger.error(f"Custom Exception  : {str(ce)}")

        finally:
            logger.info("Data Ingestion Completed.")





if __name__ == '__main__':

    ingest = DataIngestion(read_yaml(CONFIG_PATH))
    ingest.run()