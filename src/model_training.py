import os
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.paths_config import (
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH
)
from utils.common_functions import load_data
import mlflow
import mlflow.sklearn
from tensorflow import keras
from tensorflow.keras import layers, regularizers

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

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

            logger.info("Data split successfully for model training")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error during loading data: {e}")
            raise CustomException("Failed to load data", e)

    def train_ann_model(self, X_train, y_train):
        """
        Trains an improved ANN model with better regularization and layer ordering.
        """
        try:
            logger.info("Initializing improved model")

            model = keras.Sequential([
                # --- Input & First Hidden Layer ---
                # We make the model wider (more capacity)
                # We add L2 regularization to penalize large weights
                # We set use_bias=False as BatchNormalization will add its own bias (beta)
                layers.Dense(
                    256, 
                    use_bias=False, # Bias is redundant with BatchNormalization
                    input_shape=(X_train.shape[1],),
                    kernel_regularizer=regularizers.l2(0.001) # L2 Regularization
                ),
                # Apply BatchNormalization *before* the activation
                layers.BatchNormalization(),
                # Apply activation *after* BatchNormalization
                layers.Activation('relu'),
                # Increase dropout slightly for the wider layer
                layers.Dropout(0.4),

                # --- Second Hidden Layer ---
                layers.Dense(
                    128, 
                    use_bias=False, 
                    kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.4),

                # --- Third Hidden Layer ---
                layers.Dense(
                    64, 
                    use_bias=False, 
                    kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.3), # Can use a slightly lower dropout deeper in

                # --- Output Layer ---
                layers.Dense(6, activation='softmax')
            ])

            logger.info("Model initialized successfully")

            # Explicitly define the optimizer to easily tune the learning rate
            optimizer = keras.optimizers.Adam(learning_rate=1e-3) 

            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Model compiled successfully")

            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,      # Reduce LR by half
                patience=5,      # If no improvement for 5 epochs
                min_lr=1e-6      # Floor for the learning rate
            )

            logger.info("Starting model training")

            # Store the training history
            history = model.fit(
                X_train, y_train,
                validation_split=0.2, # Use 20% of training data for validation
                epochs=100,           # Increase epochs, EarlyStopping will handle it
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )

            logger.info("Model trained successfully")

            # You might want to return the history as well for plotting
            return model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")

            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1score = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1score:.4f}")

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

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                ann_model = self.train_ann_model(X_train, y_train)
                
                metrics = self.evaluate_model(ann_model, X_test, y_test)
                
                self.save_model(ann_model)

                logger.info("Logging model to MLflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging metrics to MLflow")
                mlflow.log_metrics(metrics)

                logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error while running model training pipeline: {e}")
            raise CustomException("Failed to run training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()