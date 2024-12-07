import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def preprocess_data(self, df):
        """
        Perform text preprocessing specific to NLP tasks.
        For example:
        - Removing special characters, URLs, and mentions
        - Converting text to lowercase
        - Removing stopwords, etc.
        """
        logging.info("Starting text preprocessing")
        df['cleaned_text'] = (
            df['text']
            .str.replace(r'http\S+|www.\S+', '', regex=True)  # Remove URLs
            .str.replace(r'@\w+', '', regex=True)  # Remove mentions
            .str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove special characters
            .str.lower()  # Convert to lowercase
            .str.strip()  # Remove leading and trailing spaces
        )
        logging.info("Text preprocessing completed")
        return df

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            # Load the dataset
            df = pd.read_csv(os.path.join('notebooks', 'data', 'tweets.csv'))
            logging.info('Dataset read as pandas DataFrame')

            # Ensure the dataset has the necessary columns
            if 'text' not in df.columns or 'sentiment' not in df.columns:
                raise CustomException("Dataset must contain 'text' and 'sentiment' columns")

            # Preprocess text data
            df = self.preprocess_data(df)

            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved')

            # Train-test split
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error('Error occurred in Data Ingestion Config')
            raise CustomException(e, sys)