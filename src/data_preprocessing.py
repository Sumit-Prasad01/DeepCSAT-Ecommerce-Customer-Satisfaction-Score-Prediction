import os
import numpy as np
import pandas as pd
import pickle
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        # self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)



    def data_processing(self, df: pd.DataFrame):
        try:
            logger.info("Starting our data processing step with One-Hot Encoding")
            logger.info("Dropping the columns")

            
            df = df.copy()

            df.drop(columns=['Unique id', 'connected_handling_time', 'Order_id'], inplace=True)
            df.drop_duplicates(inplace=True)

            
            self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
            
            self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.to_list()

            for col in self.cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if df[col].mode().size > 0 else 'Unknown')
            
            df['Customer Remarks'] = df['Customer Remarks'].fillna('No remarks')

            df['order_date_time'] = pd.to_datetime(df['order_date_time'], format='mixed', errors='coerce')
            df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], format='mixed', errors='coerce')
            df['issue_responded'] = pd.to_datetime(df['issue_responded'], format='mixed', errors='coerce')
            df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'], format='mixed', errors='coerce')            

            df['Item_price'] = df['Item_price'].fillna(df['Item_price'].median())

            df['Issue_reported at'] = df['Issue_reported at'].fillna(df['order_date_time'])
            median_response_time = (df['issue_responded'] - df['Issue_reported at']).median()
            df['issue_responded'] = df['issue_responded'].fillna(df['Issue_reported at'] + median_response_time)
            df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])

            df['missing_order_time'] = df['order_date_time'].isna().astype(int)
            df['missing_report_time'] = df['Issue_reported at'].isna().astype(int)
            df['missing_response_time'] = df['issue_responded'].isna().astype(int)

            median_response_time = (df['issue_responded'] - df['Issue_reported at']).median()

            df['Issue_reported at'] = df['Issue_reported at'].fillna(df['order_date_time'])
            df['issue_responded'] = df['issue_responded'].fillna(df['Issue_reported at'] + median_response_time)
            df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])


            df['response_time_hours'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 3600
            df['order_to_issue_hours'] = (df['Issue_reported at'] - df['order_date_time']).dt.total_seconds() / 3600
            df['survey_delay_hours'] = (pd.to_datetime(df['Survey_response_Date'], errors='coerce') - df['issue_responded']).dt.total_seconds() / 3600   

            df[['response_time_hours', 'order_to_issue_hours', 'survey_delay_hours']] = df[['response_time_hours', 'order_to_issue_hours', 'survey_delay_hours']].fillna(0)

            tenure_map = {
                'On Job Training': 0,
                '0-30': 1,
                '31-60': 2,
                '61-90': 3,
                '>90': 4
            }

            df['Tenure Bucket'] = df['Tenure Bucket'].map(tenure_map)

            self.label_encoders = {}  # to store encoders per column
            mappings = {}

            for col in self.cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

            skew_threshold = 5
            skewness = df[self.num_cols].apply(lambda x : x.skew()) 

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            logger.info("Data Preprocessing done.")

            return df

        except Exception as e:
            logger.error(f"Error During data preprocessing {e}")
            raise CustomException("Error while preprocessing data.", e)
        

    def balance_data(self, df : pd.DataFrame):
        try:
            logger.info("Handling Imbalanced data")
            X = df.drop(columns = 'CSAT Score')
            y = df['CSAT Score']

            smote = SMOTE(random_state = 42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            balanced_df = pd.DataFrame(X_resampled, columns = X.columns)
            balanced_df['CSAT Score'] = y_resampled

            logger.info("Data Balanced Successfully")

            return balanced_df
        
        except Exception as e:
            logger.error(f"Error During data balancing {e}")
            raise CustomException("Error while balancing data.", e)
        

    def save_data(self, df : pd.DataFrame, file_path):
        try:
            logger.info("Svaing Our data into processed folder.")

            df.to_csv(file_path, index = False)
            os.makedirs("artifacts/models", exist_ok = True)
            with open(ENCODER_OUTPUT_PATH, "wb") as f:
                pickle.dump(self.label_encoders, f)


            logger.info(f"Data Saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error During Saving Data {e}")
            raise CustomException("Error while Saving Data.", e)
    

    def process(self):
        try:
            logger.info("Loading the data from raw dir.")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.data_processing(train_df)
            test_df = self.data_processing(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing completed successfully.")
        
        except Exception as e:
            logger.error(f"Error During processing data pipeline {e}")
            raise CustomException("Error while Processing Data Pipeline.", e)
        


if __name__ == "__main__":

    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR)
    processor.process()