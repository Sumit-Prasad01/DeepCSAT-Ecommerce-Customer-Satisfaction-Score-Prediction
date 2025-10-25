import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def data_processing(self, df: pd.DataFrame):
        df = df.copy()

        columns_to_drop = ['Unique id', 'Order_id', 'order_date_time', 'Issue_reported at', 
                          'issue_responded', 'Survey_response_Date']
        
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        columns_to_drop.extend(unnamed_cols)
        
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        df.drop_duplicates(inplace=True)

        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
        self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
        
        if 'CSAT Score' in self.cat_cols:
            self.cat_cols.remove('CSAT Score')
        if 'CSAT Score' in self.num_cols:
            self.num_cols.remove('CSAT Score')

        for col in self.cat_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()[0] if df[col].mode().size > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        if 'Customer Remarks' in df.columns:
            df['Customer Remarks'] = df['Customer Remarks'].fillna('No remarks')

        if 'Item_price' in df.columns:
            df['Item_price'] = df['Item_price'].fillna(df['Item_price'].median())
        
        if 'connected_handling_time' in df.columns:
            df['connected_handling_time'] = df['connected_handling_time'].fillna(df['connected_handling_time'].median())



        if 'Tenure Bucket' in df.columns:
            tenure_map = {
                'On Job Training': 0,
                '0-30': 1,
                '31-60': 2,
                '61-90': 3,
                '>90': 4
            }
            df['Tenure Bucket'] = df['Tenure Bucket'].map(tenure_map)
            df['Tenure Bucket'] = df['Tenure Bucket'].fillna(0)

        self.label_encoders = {}
        mappings = {}

        for col in self.cat_cols:
            if col in df.columns and col != 'CSAT Score':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        skew_threshold = 5
        current_num_cols = [col for col in self.num_cols if col in df.columns]
        skewness = df[current_num_cols].apply(lambda x: x.skew())

        for column in skewness[skewness > skew_threshold].index:
            df[column] = np.log1p(df[column])
        
        return df

        

    def balance_data(self, df: pd.DataFrame):
        if 'CSAT Score' not in df.columns:
            return df
        
        X = df.drop(columns='CSAT Score')
        y = df['CSAT Score']

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df['CSAT Score'] = y_resampled

        return balanced_df
        

    def save_data(self, df: pd.DataFrame, file_path, encoder_path):
        df.to_csv(file_path, index=False)
        
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoders, f)
    

    def process(self, train_output_path, test_output_path, encoder_path):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_df = self.data_processing(train_df)
        test_df = self.data_processing(test_df)

        train_df = self.balance_data(train_df)
        test_df = self.balance_data(test_df)

        test_df = test_df[train_df.columns]

        self.save_data(train_df, train_output_path, encoder_path)
        self.save_data(test_df, test_output_path, encoder_path)


# if __name__ == "__main__":
#     processor = DataProcessor('data/raw/train.csv', 'data/raw/test.csv', 'data/processed')
#     processor.process('data/processed/train.csv', 'data/processed/test.csv', 'artifacts/models/encoders.pkl')