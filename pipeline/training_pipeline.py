from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *



if __name__ == '__main__':

    # ingest = DataIngestion(read_yaml(CONFIG_PATH))
    # ingest.run()

    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR)
    processor.process(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, ENCODER_OUTPUT_PATH)

    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()

    