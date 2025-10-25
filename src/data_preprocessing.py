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

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def data_processing(self, df: pd.DataFrame):
        try:
            logger.info("Starting data processing")
            
            df = df.copy()

            # Drop columns that are identifiers or not useful for modeling
            logger.info("Dropping identifier columns")
            columns_to_drop = ['Unique id', 'Order_id']
            
            # Also drop any unnamed index columns
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
            columns_to_drop.extend(unnamed_cols)
            
            df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
            df.drop_duplicates(inplace=True)

            # Identify categorical and numerical columns
            self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
            self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
            
            # Remove target variable from categorical columns if present
            if 'CSAT Score' in self.cat_cols:
                self.cat_cols.remove('CSAT Score')
            if 'CSAT Score' in self.num_cols:
                self.num_cols.remove('CSAT Score')

            # Handle missing values in categorical columns
            logger.info("Handling missing values in categorical columns")
            for col in self.cat_cols:
                if df[col].isna().any():
                    mode_val = df[col].mode()[0] if df[col].mode().size > 0 else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
            
            # Special handling for Customer Remarks if empty
            if 'Customer Remarks' in df.columns:
                df['Customer Remarks'] = df['Customer Remarks'].fillna('No remarks')

            # Convert datetime columns
            logger.info("Converting datetime columns")
            datetime_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')

            # Handle missing values in numerical columns
            logger.info("Handling missing values in numerical columns")
            if 'Item_price' in df.columns:
                df['Item_price'] = df['Item_price'].fillna(df['Item_price'].median())
            
            if 'connected_handling_time' in df.columns:
                df['connected_handling_time'] = df['connected_handling_time'].fillna(df['connected_handling_time'].median())

            # Handle missing datetime values with forward/backward fill logic
            if 'Issue_reported at' in df.columns and 'order_date_time' in df.columns:
                df['Issue_reported at'] = df['Issue_reported at'].fillna(df['order_date_time'])
            
            if 'issue_responded' in df.columns and 'Issue_reported at' in df.columns:
                median_response_time = (df['issue_responded'] - df['Issue_reported at']).median()
                df['issue_responded'] = df['issue_responded'].fillna(df['Issue_reported at'] + median_response_time)
            
            if 'order_date_time' in df.columns and 'Issue_reported at' in df.columns:
                df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])

            # Map Tenure Bucket to ordinal values
            logger.info("Mapping Tenure Bucket to ordinal values")
            if 'Tenure Bucket' in df.columns:
                tenure_map = {
                    'On Job Training': 0,
                    '0-30': 1,
                    '31-60': 2,
                    '61-90': 3,
                    '>90': 4
                }
                df['Tenure Bucket'] = df['Tenure Bucket'].map(tenure_map)
                # Handle any unmapped values
                df['Tenure Bucket'] = df['Tenure Bucket'].fillna(0)

            # Label encode categorical columns
            logger.info("Label encoding categorical columns")
            self.label_encoders = {}
            mappings = {}

            for col in self.cat_cols:
                if col in df.columns and col != 'CSAT Score':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

            # Handle skewness in numerical columns
            logger.info("Handling skewness in numerical columns")
            skew_threshold = 5
            current_num_cols = [col for col in self.num_cols if col in df.columns]
            skewness = df[current_num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            logger.info("Data preprocessing completed")
            return df

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Error while preprocessing data.", e)
        

    def balance_data(self, df: pd.DataFrame):
        try:
            logger.info("Handling imbalanced data with SMOTE")
            
            if 'CSAT Score' not in df.columns:
                logger.warning("CSAT Score column not found, skipping balancing")
                return df
            
            X = df.drop(columns='CSAT Score')
            y = df['CSAT Score']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['CSAT Score'] = y_resampled

            logger.info("Data balanced successfully")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during data balancing: {e}")
            raise CustomException("Error while balancing data.", e)
        

    def save_data(self, df: pd.DataFrame, file_path):
        try:
            logger.info(f"Saving data to {file_path}")

            df.to_csv(file_path, index=False)
            
            # Save label encoders
            os.makedirs("artifacts/models", exist_ok=True)
            with open(ENCODER_OUTPUT_PATH, "wb") as f:
                pickle.dump(self.label_encoders, f)

            logger.info(f"Data saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data: {e}")
            raise CustomException("Error while saving data.", e)
    

    def process(self):
        try:
            logger.info("Loading data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Processing training data")
            train_df = self.data_processing(train_df)
            
            logger.info("Processing test data")
            test_df = self.data_processing(test_df)

            logger.info("Balancing training data")
            train_df = self.balance_data(train_df)
            
            logger.info("Balancing test data")
            test_df = self.balance_data(test_df)

            # Ensure test data has same columns as train data
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully")
        
        except Exception as e:
            logger.error(f"Error during processing data pipeline: {e}")
            raise CustomException("Error while processing data pipeline.", e)


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR)
    processor.process()