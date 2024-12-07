import os
import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    try:
        # Step 1: Data Ingestion (In this case, tweets dataset)
        logging.info("Data Ingestion started")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f"Train and Test data paths: {train_data_path}, {test_data_path}")

        # Step 2: Data Transformation (Text preprocessing: tokenization, stopword removal, vectorization)
        logging.info("Data Transformation started")
        data_transformation = DataTransformation()

        # Assuming the DataTransformation class takes care of cleaning the tweet text
        # and transforming it into numerical vectors (e.g., using TfidfVectorizer)
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Step 3: Model Training (Train sentiment analysis model using the transformed data)
        logging.info("Model Training started")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise CustomException(e, sys)
