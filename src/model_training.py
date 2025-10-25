import os
import numpy as np
import pickle
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from config.paths_config import (
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH,
)
from utils.common_functions import load_data
import mlflow
import mlflow.sklearn
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.scaler = StandardScaler()

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['CSAT Score'])
            y_train = train_df['CSAT Score']

            X_test = test_df.drop(columns=['CSAT Score'])
            y_test = test_df['CSAT Score']

            # Check the unique values in target variable
            logger.info(f"Unique CSAT Score values in train: {sorted(y_train.unique())}")
            logger.info(f"Unique CSAT Score values in test: {sorted(y_test.unique())}")
            logger.info(f"Train class distribution:\n{y_train.value_counts().sort_index()}")

            # Convert CSAT Score from 1-5 to 0-4 for model compatibility
            y_train = y_train - 1
            y_test = y_test - 1

            logger.info("Data split successfully for model training")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error during loading data: {e}")
            raise CustomException("Failed to load data", e)

    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler - critical for neural networks
        """
        try:
            logger.info("Scaling features using StandardScaler")
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Feature scaling completed")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}")
            raise CustomException("Failed to scale features", e)

    def compute_class_weights(self, y_train):
        """
        Compute class weights to handle any remaining class imbalance
        """
        try:
            logger.info("Computing class weights")
            
            classes = np.unique(y_train)
            class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            
            class_weights_dict = dict(zip(classes, class_weights_array))
            logger.info(f"Class weights: {class_weights_dict}")
            
            return class_weights_dict
            
        except Exception as e:
            logger.error(f"Error computing class weights: {e}")
            return None

    def train_ann_model(self, X_train, y_train, X_val, y_val, class_weights=None):
        """
        Trains an improved ANN model optimized for CSAT prediction
        """
        try:
            logger.info("Initializing ANN model")
            
            num_classes = len(np.unique(y_train))
            input_dim = X_train.shape[1]
            
            logger.info(f"Input dimension: {input_dim}")
            logger.info(f"Number of classes: {num_classes}")

            model = keras.Sequential([
                # Input & First Hidden Layer
                layers.Dense(
                    256, 
                    use_bias=False,
                    input_shape=(input_dim,),
                    kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.4),

                # Second Hidden Layer
                layers.Dense(
                    128, 
                    use_bias=False, 
                    kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.4),

                # Third Hidden Layer
                layers.Dense(
                    64, 
                    use_bias=False, 
                    kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.3),

                # Output Layer - dynamic based on number of classes
                layers.Dense(num_classes, activation='softmax')
            ])

            logger.info("Model initialized successfully")

            optimizer = keras.optimizers.Adam(learning_rate=1e-3) 

            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Model compiled successfully")

            # Callbacks
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )

            logger.info("Starting model training")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                class_weight=class_weights,
                verbose=1
            )

            logger.info("Model trained successfully")

            return model, history

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")

            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1score = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1score:.4f}")

            # Additional detailed metrics
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_matrix(y_test, y_pred)))

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1score
            }

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Failed to evaluate model", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving model")
            model.save(self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")

            # Save scaler
            scaler_path = self.model_output_path.replace('.keras', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")

        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting MLflow experiment")
                logger.info("Starting model training pipeline")

                logger.info("Logging datasets to MLflow")
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                # Load data
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                # Scale features
                X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
                
                # Create validation split from training data
                from sklearn.model_selection import train_test_split
                X_train_final, X_val, y_train_final, y_val = train_test_split(
                    X_train_scaled, y_train, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=y_train
                )
                
                logger.info(f"Training set size: {X_train_final.shape[0]}")
                logger.info(f"Validation set size: {X_val.shape[0]}")
                logger.info(f"Test set size: {X_test_scaled.shape[0]}")
                
                # Compute class weights
                class_weights = self.compute_class_weights(y_train_final)
                
                # Train model
                ann_model, history = self.train_ann_model(
                    X_train_final, y_train_final, 
                    X_val, y_val,
                    class_weights
                )
                
                # Evaluate model
                metrics = self.evaluate_model(ann_model, X_test_scaled, y_test)
                
                # Save model and scaler
                self.save_model(ann_model)

                logger.info("Logging model to MLflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging metrics to MLflow")
                mlflow.log_metrics(metrics)
                
                # Log hyperparameters
                mlflow.log_params({
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "batch_size": 64,
                    "epochs": 150,
                    "early_stopping_patience": 15,
                    "dropout_rate": "0.4, 0.4, 0.3",
                    "l2_regularization": 0.001,
                    "architecture": "256-128-64"
                })

                logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error while running model training pipeline: {e}")
            raise CustomException("Failed to run training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()