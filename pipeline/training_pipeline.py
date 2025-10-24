from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from utils.common_functions import read_yaml
from config.paths_config import *



if __name__ == '__main__':

    ingest = DataIngestion(read_yaml(CONFIG_PATH))
    ingest.run()


    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR)
    processor.process()